import requests

CLIMATIQ_API_KEY = "TZM4HEYWE979F0GG62PNMP7KBG"
DATA_VERSION = "20"

headers = {
    "Authorization": f"Bearer {CLIMATIQ_API_KEY}"
}

def get_fuzzy_search_cfactors(query:str):

    data_input = {
        'data_version' : DATA_VERSION,
        'query' : query
    }

    r = requests.get(
        "https://api.climatiq.io/data/v1/search",
        headers=headers,
        params=data_input
    )
    
    print(r.json())
    results = r.json()["results"]


    if r.status_code == 200:
        return results
    else:
        print("Failed to retrieve fuzzy carbon emission factor search results")
        print(f"Status code: {r.status_code}")
        print(f"Response: {r.json()}")
        return None

def estimate_carbon_emission(
        activity_id:str, 
        region:str,
        activity_size:int,
        activity_unit:str
    ):

    data_input = {
        "emission_factor": {
            "activity_id": activity_id,
            "data_version": DATA_VERSION,
            "region": region
        },
        "parameters": {
            "distance": activity_size,
            "distance_unit": activity_unit
        }
    }


    r = requests.post(
        "https://api.climatiq.io/data/v1/estimate",
        headers=headers,
        json = data_input
    )

    result = r.json()

    if r.status_code == 200:
        return result
    else:
        print("Failed to estimate carbon emission")
        print(f"Status code: {r.status_code}")
        print(f"Response: {r.json()}")
        return None
    

def suggest_carbon_emission_factors(
        input_text:str, 
        domain_text:str
    ):

    data_input = {
        "suggest": {
            "text": input_text,
            "domain": domain_text
        },
        "max_suggestions" : 5
    }


    r = requests.post(
        "https://preview.api.climatiq.io/autopilot/v1-preview3/suggest",
        headers=headers,
        json = data_input
    )

    result = r.json()

    if r.status_code == 200:
        return result
    else:
        print("Failed to suggest carbon emission factors")
        print(f"Status code: {r.status_code}")
        print(f"Response: {r.json()}")
        return None


def main():

    query = "wind electricity"

    carbon_emission_factors = get_fuzzy_search_cfactors(query=query)

    for fact in carbon_emission_factors:
        print(fact["region"])

    # activity_id = "passenger_vehicle-vehicle_type_car-fuel_source_bev-engine_size_gt_2000cc_lt_3000cc-vehicle_age_post_2015-vehicle_weight_na"

    # region = "NZ"

    # activity_size = 10

    # activity_unit = "km"

    # emission_estimate = estimate_carbon_emission(
    #     activity_id=activity_id,
    #     region=region,
    #     activity_size=activity_size,
    #     activity_unit=activity_unit
    # )

    # print(emission_estimate)

    # input_text = "building a house"
    # domain_text = "general"

    # carbon_emission_factors = suggest_carbon_emission_factors(
    #     input_text=input_text,
    #     domain_text=domain_text
    # )

    # print(carbon_emission_factors)


if __name__ == "__main__":
    main()