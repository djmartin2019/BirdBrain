# Birdbrain web

SvelteKit frontend for [birdbrain.djm-apps.com](https://birdbrain.djm-apps.com).

## Routes

| Path | Page |
|------|------|
| `/` | Upload + identify (calls `/api/predict`) |
| `/species` | Searchable list of 200 CUB-200 species the models recognize |
| `/about` | Project overview |
| `/docs` | User docs + links to repo `docs/` |
| `/citation` | CUB-200-2011 citation |

## Development

```bash
npm install
npm run dev
```

Open http://localhost:5173

## Build

```bash
npm run build
npm run preview
```

Static output is written to `build/` (adapter-static, all routes prerendered).

## Environment

Copy `.env.example` to `.env` if the API is hosted on a different origin in production:

```
VITE_API_URL=
```

Leave empty when nginx serves the API under the same host at `/api`.
