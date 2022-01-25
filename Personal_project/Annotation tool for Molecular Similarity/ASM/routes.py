import os
import datetime
import json
from datetime import datetime
import random

from sqlalchemy import func
import time as Time

from werkzeug.utils import redirect
# from ASM.forms import *
from ASM.database import customer, administrator
from ASM import app, db, Config, email
from flask import render_template, flash, jsonify, session, url_for, request
from werkzeug.security import check_password_hash, generate_password_hash
from ASM.models import *

@app.errorhandler(404)
def page_not_fount(e):
    return render_template('customer/404-illustration.html')

@app.route('/index',methods=['POST','GET'])
def index():
    return render_template('index.html')