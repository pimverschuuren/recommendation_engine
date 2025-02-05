from flask import Flask, jsonify, request
from pydantic import BaseModel, ValidationError

# Define a Pydantic model for incoming requests
class RecommendationRequest(BaseModel):
    input_text: str


class RecommendationResponse(BaseModel):
    recommendation : str

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Recommend a product!</p>"

@app.route('/get_recommendation', methods=["POST"])
def get_recommendation():

    try:
        request_input = RecommendationRequest(**request.get_json())
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    
    # Save the user to the database, etc.
    recommendation = request_input.input_text + " product"

    recommendation_response = RecommendationResponse(
        recommendation = recommendation
    )

    # Return a JSON response with the user data
    return jsonify(recommendation_response.dict())

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)