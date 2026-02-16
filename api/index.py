from api.app import app

# Vercel expects the Flask app to be available as 'app'
# This file serves as the entry point for Vercel
if __name__ == "__main__":
    app.run()