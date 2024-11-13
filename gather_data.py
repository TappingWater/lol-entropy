import requests
import time
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# RIOT API Parameters
API_KEY = 'RGAPI-2af01446-34d2-47c3-b6a5-1d21604af28d'  # Replace with your actual API key
HEADERS = {'X-Riot-Token': API_KEY}
REGION = 'americas'
PLATFORM = 'na1'

processed_puuids = set()
processed_matches = set()
dataset = []

DESIRED_DATASET_SIZE = 100
RATE_LIMIT_WAIT_TIME = 1.2

def get_puuid(game_name, tag_line):
    url = f"https://{REGION}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}?api_key={API_KEY}"
    logging.info(f"Requesting PUUID for {game_name}#{tag_line} from URL: {url}")
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        puuid = data.get('puuid')
        logging.info(f"Retrieved PUUID for {game_name}#{tag_line}: {puuid}")
        return puuid
    else:
        logging.error(f"Failed to retrieve PUUID for {game_name}#{tag_line}. HTTP Status Code: {response.status_code}")
        logging.error(f"API Response: {response.text}")
        return None

def get_match_ids(puuid):
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count=20&api_key={API_KEY}"
    response = requests.get(url, headers=HEADERS)
    match_ids = response.json()
    if isinstance(match_ids, list):
        logging.info(f"Retrieved {len(match_ids)} match IDs for PUUID: {puuid}")
    else:
        logging.warning(f"No matches found for PUUID: {puuid}")
        match_ids = []
    return match_ids

def get_match_data(match_id):
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={API_KEY}"
    response = requests.get(url, headers=HEADERS)
    match_data = response.json()
    if match_data:
        logging.info(f"Retrieved data for match ID: {match_id}")
    else:
        logging.error(f"Failed to retrieve data for match ID: {match_id}")
    return match_data

def get_champion_mastery(puuid, champion_id):
    url = f"https://{PLATFORM}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{champion_id}?api_key={API_KEY}"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    mastery_level = data.get('championLevel')
    if mastery_level is None:
        logging.warning(f"No mastery data for PUUID: {puuid}, Champion ID: {champion_id}")
    return mastery_level

def process_match(match_data, target_puuid):
    info = match_data.get('info', {})
    participants = info.get('participants', [])
    if not participants:
        logging.warning(f"No participant data in match: {match_data.get('metadata', {}).get('matchId')}")
        return

    for participant in participants:
        if participant['puuid'] == target_puuid:
            # Extract features for the target player
            user_champion = participant['championName']
            user_champion_id = participant['championId']
            user_win = participant['win']
            encrypted_summoner_id = participant['summonerId']
            user_champion_mastery = get_champion_mastery(target_puuid, user_champion_id)

            # Get ally and enemy champions and masteries
            ally_champions = []
            ally_masteries = []
            enemy_champions = []
            enemy_masteries = []
            for p in participants:
                if p['puuid'] != target_puuid:
                    if p['teamId'] == participant['teamId']:
                        # Ally
                        ally_champions.append(p['championName'])
                        mastery = get_champion_mastery(p['puuid'], p['championId'])
                        ally_masteries.append(mastery)
                    else:
                        # Enemy
                        enemy_champions.append(p['championName'])
                        mastery = get_champion_mastery(p['puuid'], p['championId'])
                        enemy_masteries.append(mastery)

            # Ensure we have the correct number of allies and enemies
            if len(ally_champions) != 4 or len(enemy_champions) != 5:
                logging.warning(f"Unexpected number of allies or enemies in match: {match_data.get('metadata', {}).get('matchId')}")
                return

            # Prepare the data dictionary
            data_entry = {
                'win': int(user_win),
                'user_champion': user_champion,
                'user_champion_mastery': user_champion_mastery,
            }

            # Add ally data
            for i in range(4):
                data_entry[f'ally{i+1}_champion'] = ally_champions[i]
                data_entry[f'ally{i+1}_mastery'] = ally_masteries[i]

            # Add enemy data
            for i in range(5):
                data_entry[f'enemy{i+1}_champion'] = enemy_champions[i]
                data_entry[f'enemy{i+1}_mastery'] = enemy_masteries[i]

            # Add to dataset
            dataset.append(data_entry)
            logging.info(f"Processed match for PUUID {target_puuid}")
            break  # Only process the target player

def collect_data(start_puuid):
    puuid_queue = [start_puuid]

    while puuid_queue and len(dataset) < DESIRED_DATASET_SIZE:
        puuid = puuid_queue.pop(0)
        if puuid in processed_puuids:
            continue
        processed_puuids.add(puuid)
        logging.info(f"Processing PUUID: {puuid}")

        try:
            match_ids = get_match_ids(puuid)
            for match_id in match_ids:
                if match_id in processed_matches:
                    continue
                processed_matches.add(match_id)

                match_data = get_match_data(match_id)
                process_match(match_data, puuid)

                # Extract participant PUUIDs and add to queue
                participants_puuids = match_data.get('metadata', {}).get('participants', [])
                for participant_puuid in participants_puuids:
                    if participant_puuid not in processed_puuids:
                        puuid_queue.append(participant_puuid)

                # Handle rate limits
                time.sleep(RATE_LIMIT_WAIT_TIME)

                if len(dataset) >= DESIRED_DATASET_SIZE:
                    break
        except Exception as e:
            logging.error(f"Error processing PUUID {puuid}: {e}")
            time.sleep(RATE_LIMIT_WAIT_TIME)  # Wait before retrying

# Preprocessing Functions
def discretize_mastery(mastery_level):
    if mastery_level is None:
        return 'Unknown'  # Handle missing values
    else:
        try:
            mastery_level = int(mastery_level)
            if mastery_level <= 2:
                return 'Low'
            elif mastery_level <= 5:
                return 'Medium'
            else:
                return 'High'
        except ValueError:
            return 'Unknown'

def preprocess_data(df):
    # Discretize mastery levels
    mastery_cols = ['user_champion_mastery'] + [f'ally{i}_mastery' for i in range(1,5)] + [f'enemy{i}_mastery' for i in range(1,6)]
    for col in mastery_cols:
        df[col] = df[col].apply(discretize_mastery)

    # Handle any missing champion names
    champion_cols = ['user_champion'] + [f'ally{i}_champion' for i in range(1,5)] + [f'enemy{i}_champion' for i in range(1,6)]
    for col in champion_cols:
        df[col] = df[col].fillna('Unknown')

    # Ensure all data is in string format
    df = df.astype(str)

    # Return the preprocessed DataFrame
    return df

if __name__ == '__main__':
    game_name = 'Davemon'
    tag_line = 'NA1'
    start_puuid = get_puuid(game_name, tag_line)
    if start_puuid:
        collect_data(start_puuid)

        # Convert the dataset to a DataFrame
        df = pd.DataFrame(dataset)

        # Preprocess the data
        logging.info("Starting data preprocessing...")
        df = preprocess_data(df)
        logging.info("Data preprocessing completed.")

        # Save the dataset to a CSV file
        df.to_csv('league_dataset.csv', index=False)
        logging.info(f"Dataset saved to 'league_dataset.csv' with {len(df)} records.")
    else:
        logging.error("Failed to start data collection due to missing PUUID.")
