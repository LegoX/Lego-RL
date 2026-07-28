# Harbor RL Training Dashboard

Real-time web dashboard for monitoring RL training runs. Displays training metrics, validation scores, reward curves, policy stats, and more by parsing log files and optionally connecting to wandb.

## Quick Start

```bash
cd webui
./start_dashboard.sh
```

This will build the frontend (if needed), start the API server on port 8090, and optionally open a Cloudflare tunnel for remote access.

## Manual Setup

### 1. Install frontend dependencies

```bash
cd webui
npm install
```

### 2. Build the frontend

```bash
npm run build
```

This produces a `dist/` directory with the static frontend.

### 3. Start the backend server

```bash
python server.py \
  --port 8090 \
  --log-dir ../logs \
  --static-dir dist
```

The dashboard is now available at `http://localhost:8090`.

### 4. (Optional) Enable wandb integration

```bash
python server.py \
  --port 8090 \
  --log-dir ../logs \
  --static-dir dist \
  --wandb-entity YOUR_ENTITY \
  --wandb-project YOUR_PROJECT \
  --wandb-api-key YOUR_KEY
```

Or set environment variables: `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_API_KEY`.

## Development Mode

Run the frontend dev server (with hot reload) and the backend API server separately:

```bash
# Terminal 1: API server
python server.py --port 8080 --log-dir ../logs

# Terminal 2: Frontend dev server (proxies /api to :8080)
npm run dev
```

The dev frontend runs at `http://localhost:3000`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOG_DIR` | `../logs` | Directory containing `.log` / `.out` training log files |
| `PORT` | `8090` | Server port |
| `TUNNEL` | `true` | Whether to open a Cloudflare tunnel (`start_dashboard.sh` only) |
| `WANDB_ENTITY` | | wandb entity name |
| `WANDB_PROJECT` | `swe-lego-live-rl` | wandb project name |
| `WANDB_API_KEY` | | wandb API key |

## Server CLI Options

```
python server.py [OPTIONS]

  --port PORT              Server port (default: 8080)
  --host HOST              Bind address (default: 0.0.0.0)
  --log-dir DIR            Primary log directory
  --extra-log-dir DIR      Additional log directories (repeatable)
  --static-dir DIR         Directory with built frontend (default: dist/)
  --wandb-entity ENTITY    wandb entity
  --wandb-project PROJECT  wandb project
  --wandb-api-key KEY      wandb API key
```

## Dashboard Panels

- **Overview** - Key metrics at a glance with sparklines
- **Rewards & Scores** - Reward signals, advantages, returns
- **Policy & Training** - Actor loss, entropy, KL, clip fraction
- **Agent Loop** - Agent execution timing breakdown
- **Validation** - Validation reward scores and turn statistics
- **Sequences** - Response/prompt lengths
- **Performance** - Throughput, step time, MFU
- **Stability** - Gradient norms, rollout correlation
- **AI Analysis** - LLM-powered training analysis
- **Logs** - Raw log file viewer
- **Explorer** - Custom metric explorer
