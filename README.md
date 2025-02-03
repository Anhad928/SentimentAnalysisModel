# AI Sentiment Analysis Using Friends Web-Series Scene Cuts

This repository contains a full-stack project for training an AI model to perform sentiment analysis using scene cuts from the Friends TV show. The project leverages various AWS tools for data storage, processing, and deployment. The AI model analyzes the sentiment of characters' dialogues, providing insights into the emotional tone of different scenes.

---

## Project Overview

This project aims to:

1. Extract dialogues and scene cuts from the Friends web series.
2. Preprocess and label the data for sentiment analysis.
3. Train a machine learning model using labeled data to classify sentiments (e.g., positive, neutral, negative).
4. Deploy the trained model to a web application that allows users to upload scenes and get sentiment predictions.
5. Use AWS tools for efficient data storage, model training, and deployment.

---

## Features

- **Scene Cut Extraction**: Extract and preprocess dialogue from video files.
- **Sentiment Analysis**: Classify the sentiment of scenes using a trained AI model.
- **Web Interface**: User-friendly web application for uploading scene clips and displaying results.
- **Cloud Integration**: Leverages AWS for storage, training, and deployment.

---

## Tech Stack

### Backend:
- Python
- Flask
- TensorFlow/Keras for model training

### Frontend:
- React.js
- Tailwind CSS

### Database:
- PostgreSQL

---

## AWS Tools Used

- **AWS S3**: Storage for video files and preprocessed data.
- **AWS Lambda**: For processing scene cuts and running preprocessing scripts.
- **AWS SageMaker**: For model training and deployment.
- **AWS RDS**: PostgreSQL database for storing labeled data and results.
- **AWS CloudFront**: For serving the web application securely.
- **AWS API Gateway**: For managing API requests between the frontend and backend.

---

## Architecture

1. **Data Pipeline**:
   - Video files are uploaded to an S3 bucket.
   - AWS Lambda preprocesses and extracts scene cuts and dialogues.

2. **Model Training**:
   - Labeled data is stored in RDS.
   - SageMaker trains a sentiment analysis model on the dataset.

3. **Web Application**:
   - Frontend built with React.js for user interaction.
   - Backend Flask API handles requests and serves predictions.

4. **Deployment**:
   - The model is deployed as an endpoint in SageMaker.
   - The web app is hosted using AWS CloudFront.

---

## Setup and Installation

### Prerequisites

- Python 3.8+

- AWS CLI configured with proper permissions

## Usage

1. Upload a video clip from the Friends web series via the web application.
2. The backend processes the clip, extracts dialogues, and runs sentiment analysis.
3. The results, including sentiment predictions for each dialogue, are displayed on the frontend.

---

## Dataset

- **Source**: Scene cuts and dialogues are sourced from the Friends TV show.
- **Preprocessing**:
  - Extracted dialogues are labeled with sentiment scores using a semi-supervised approach.
  - Data is stored in AWS RDS for training.

---

## Model Training

1. Preprocessed data is fetched from the RDS database.
2. Training occurs in SageMaker using TensorFlow/Keras.
3. The trained model is evaluated for accuracy and deployed as an API endpoint.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push the branch:
   ```bash
   git push origin feature-name
   ```
4. Create a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

