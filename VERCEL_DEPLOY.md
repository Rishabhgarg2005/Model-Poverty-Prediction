# Vercel Deployment Guide

## Deployment Steps

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   vercel
   ```

## Configuration Details

- **Framework**: Flask
- **Build Command**: None (serverless functions)
- **Output Directory**: api/
- **Install Command**: pip install -r requirements.txt
- **Python Runtime**: 3.11

## Environment Variables

No environment variables are required for basic deployment.

## File Structure for Vercel

```
project/
├── api/
│   ├── app.py          # Main Flask application
│   └── index.py        # Entry point
├── static/             # Static files
├── templates/          # HTML templates
├── *.pkl               # Model files
├── vercel.json         # Vercel configuration
└── requirements.txt    # Python dependencies
```

## Troubleshooting

1. **Model files not found**: Ensure .pkl files are in the root directory
2. **Static files not loading**: Check vercel.json routes configuration
3. **Import errors**: Verify PYTHONPATH is set correctly in vercel.json
4. **Timeout errors**: Increase maxDuration in vercel.json functions config

## Testing Locally

```bash
cd api
python app.py
```

Then visit http://localhost:5000