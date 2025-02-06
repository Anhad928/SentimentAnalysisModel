# Friends Sentiment Analysis Model Training

Friends Sentiment Analysis Model Training is dedicated to building and training an AI-driven sentiment analysis model using AWS SageMaker and PyTorch. This model is trained on the MELD dataset—with scene cuts from the iconic *Friends* web series—to capture and analyze nuanced emotions. The training pipeline is designed for scalability on AWS Cloud and uses Prisma with SQLite for logging and experiment tracking in development.

This repository is part of a larger full-stack project. For the user-facing application, please refer to our [Friends Sentiment Analyzer Frontend](link).

## Features
- **AI Model Training:**  
  Train a state-of-the-art sentiment analysis model using AWS SageMaker and PyTorch.
- **Dataset:**  
  Utilize the MELD dataset featuring scene cuts from *Friends*.
- **Scalable Infrastructure:**  
  Leverage AWS Cloud for scalable model training.
- **Experiment Tracking:**  
  Log experiments with Prisma and SQLite (for development purposes).

## Tech Stack
- **Machine Learning:**  
  AWS SageMaker, PyTorch
- **Dataset:**  
  MELD dataset (Friends scene cuts)
- **Logging & Experimentation:**  
  Prisma, SQLite
- **Cloud:**  
  AWS Cloud

## Architecture
This repository focuses on the training pipeline:
- **Data Processing:**  
  Preprocess and manage the MELD dataset.
- **Model Training:**  
  Train the sentiment analysis model on AWS SageMaker.
- **Experiment Logging:**  
  Store training logs and experiment data using Prisma and SQLite.
- **Integration:**  
  The trained model is later integrated with the frontend application.

## Installation & Setup

### Prerequisites
- AWS account with SageMaker access
- Node.js (for running any Node-based scripts)
- Python environment with PyTorch and AWS SDK installed
- Git

### Steps
1. **Clone the Repository:**
Install Node.js Dependencies:

bash
Copy
npm install
Configure Environment Variables: Create a .env file in the root with your AWS credentials and other configurations. For example:

env
Copy
DATABASE_URL="sqlite:./dev.db"
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
NODE_ENV=development
Set Up Python Environment: Create a virtual environment and install required Python packages (e.g., PyTorch, boto3, sagemaker):

bash
Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Run Prisma Migrations (if applicable):

bash
Copy
npx prisma migrate dev --name init
Usage
Model Training:
Execute the training scripts to start model training on AWS SageMaker.
Monitoring:
Monitor training progress using the AWS SageMaker dashboard.
Experiment Tracking:
Review logs and experiment data stored locally via Prisma and SQLite.
Frontend Integration
This model training repository is part of a complete full-stack solution. The user-facing frontend application is maintained separately.
Check out the Friends Sentiment Analyzer Frontend for the web interface that interacts with this model.

Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and open a pull request with your changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or collaboration, please contact:

Name: Anhadpreet Singh
Email: anhadpre@ualberta.ca
GitHub: @Anhad928
