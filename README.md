# AutoGen Project

This project demonstrates how to use AutoGen with NVIDIA Nemotron models via OpenAI-compatible API.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables in `.env`:
   ```
   NVIDIA_API_KEY=your_nvidia_api_key
   NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
   ```

## Project Structure

```
├── workflows/     # AutoGen workflow definitions
├── tools/         # Custom tools for agents
├── main.py        # Entry point
├── requirements.txt
├── .env           # Environment variables (not tracked)
└── .gitignore
```

## Running

```bash
python main.py
```