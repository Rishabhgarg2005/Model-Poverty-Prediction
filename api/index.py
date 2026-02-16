from api.app import app

def handler(request):
    return app(request.environ, lambda *args: None)