from vectorizer import app

from flask_restful import Resource, Api

api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world this is my first test'}

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True)

# @app.route('/')
# @app.route('/index')
# def index():
#   return "Hello, World!"
