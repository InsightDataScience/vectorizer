from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class ReusableForm(Form):
    sentences = TextField('Text:', validators=[validators.Length(min=10, max=500)])

    @app.route("/", methods=['GET', 'POST'])
    def hello():
        form = ReusableForm(request.form)


        print
        form.errors
        if request.method == 'POST':
            sentences = request.form['sentences']
            print
            sentences

        if form.validate():
            # Save the comment here.
            flash('Here is  your input: ' + sentences)
        else:
            flash('Error: One or more sentences with 10 and 500 words are required. ')

        return render_template('hello.html', form=form)


if __name__ == "__main__":
    app.run()
