#! /usr/bin/env python

"""This is a tutorial from https://github.com/InsightDataScience/data-engineering-ecosystem/wiki/Flask
TODO: might use tornado when scaling
"""

from tornado.wsgi import WSGIContainer
from tornado.ioloop import IOLoop
from tornado.web import FallbackHandler, RequestHandler, Application
from vectorizer import app

class MainHandler(RequestHandler):
 def get(self):
   self.write("This message comes from Tornado ^_^")

tr = WSGIContainer(app)

application = Application([
(r"/tornado", MainHandler),
(r".*", FallbackHandler, dict(fallback=tr)),
])

if __name__ == "__main__":
 application.listen(80)
 IOLoop.instance().start()
