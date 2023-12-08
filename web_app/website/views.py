from flask import Blueprint, render_template, request, redirect, url_for, jsonify
from flask_login import login_required, logout_user, current_user
import random
import numpy as np
import pandas as pd
from .xgboost_web import xgboost_func
from .exp_smooth_web import expsmooth_func
from .arima_web import arima_func    
from .uni_test import uni    
from .synth_test import synth    
from .hybrid_web import hybrid

views = Blueprint('views', __name__)

@views.route('/home', methods=['GET', 'POST'])
@login_required
def home():

    def rdf():
         return random.randint(1, 50)

    def data_to_np_array(data):
        data_list = data.rsplit("\"")
        data_list.pop(0)
        data_list.pop(-1)
        for i in range(data_list.count(',')):
            data_list.remove(',')
        for x in data_list:
            x = int(x)
        return np.array(data_list)

    if request.method == 'POST':
       logout_user()
       return redirect(url_for('auth.login'))
  
    if request.is_json:
        data =  pd.DataFrame(data={'arrivals':data_to_np_array(request.args.get('data'))})
        data["arrivals"] = pd.to_numeric(data["arrivals"])
        
        prediction1 = xgboost_func(data[10:]).tolist()
        prediction2 = expsmooth_func(data).tolist()
        prediction3 = arima_func(data).tolist()
        prediction4 = uni(data)[:,0].tolist()
        prediction5 = synth(data)[:,0].tolist()
        prediction6 = hybrid(data)[:,0].tolist()
        return jsonify({'prediction1': prediction1, 'prediction2': prediction2, 'prediction3': prediction3, 
            'prediction4': prediction4, 'prediction5': prediction5, 'prediction6': prediction6})
    
    return render_template("home.html", user=current_user)
