from flask import Blueprint, redirect, render_template, request, flash, url_for
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from flask_login import login_user, login_required, logout_user, current_user

auth = Blueprint('auth', __name__)

@auth.route('/', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        if 'register' in request.form:
            username = request.form.get('registerUsername')
            email = request.form.get('registerEmail')
            password = request.form.get('registerPassword')
            confirm_password = request.form.get('registerConfirmPassword')

            user = User.query.filter_by(username=username).first()
            if user:
                flash('Username already exists.', category='error')
            elif len(email) < 4:
                flash('Email must be greater than 3 characters.', category='error')
            elif len(username) < 2:
                flash('Username must be greater than 1 character.', category='error')
            elif password != confirm_password:
                flash('Passwords don\'t match.', category='error')
            elif len(password) < 7:
                flash('Password must be greater than 6 characters.', category='error')
            else:
                new_user = User(email=email, username=username, password=generate_password_hash(password, method='sha256'))
                db.session.add(new_user)
                db.session.commit()
                flash('Account created succesfully, you can now log in!', category='success')
        else:
            username = request.form.get('loginUsername')
            password = request.form.get('loginPassword')

            user = User.query.filter_by(username=username).first()
            if user:
                if check_password_hash(user.password, password):
                    login_user(user, remember=True)
                    return redirect(url_for('views.home'))
                else:
                    flash('Incorrect password, try again.', category='error')
            else:
                flash('User does not exist.', category='error')

    return render_template("login.html", user=current_user)