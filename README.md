# Text to Podcast API Server

This FastAPI server converts text files into podcast-style audio using OpenAI for script generation and text-to-speech services.

## Features

- Upload text files
- Convert text to conversational podcast scripts
- Generate audio with alternating voices
- Background processing with job status tracking
- Ready for deployment to fly.io

## Local Development

### Prerequisites

- Python 3.9+
- pip

### Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your configuration:
   ```
   DEBUG=True
   PORT=8000
   UPLOAD_DIR=./uploads
   AUDIO_DIR=./audio_output
   ```

### Running the Server

```bash
python server.py
```

The server will be available at http://localhost:8000

## API Endpoints

- `GET /` - Health check
- `POST /upload` - Upload a text file
- `POST /generate` - Generate podcast audio from a text file
- `GET /status/{job_id}` - Check the status of a generation job
- `GET /audio/{audio_id}` - Download generated audio
- `GET /voices` - Get available voice options

## Deployment to fly.io

1. Install the flyctl CLI: https://fly.io/docs/hands-on/install-flyctl/
2. Log in to fly.io:
   ```
   flyctl auth login
   ```
3. Deploy the application:
   ```
   flyctl deploy
   ```

### Setting API Keys on Fly.io

For the server to function correctly when deployed, you need to set your OpenAI API key as a secret in your Fly.io app dashboard.

1.  Go to your app's secrets page: `https://fly.io/apps/{YOUR_APP_NAME}/secrets` (replace `{YOUR_APP_NAME}` with the name of your Fly.io app, e.g., `post2podcast`).
2.  Add a new secret with the name `OPENAI_API_KEY` and paste your OpenAI API key as the value.

The server will read this secret at runtime.

## Integration with WordPress

This server is designed to work with the Post2Podcast WordPress plugin. The plugin sends text content to this API and receives podcast audio files in return.

## License

This project is licensed under the MIT License.
