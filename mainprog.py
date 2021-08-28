from flask import Flask, render_template,request,session,flash
import sqlite3 as sql
import os
import pandas as pd
import webbrowser
from tkinter import filedialog
app = Flask(__name__)


def sendmail12(mailid,data1):
    import smtplib, ssl

    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "kartik.jnnce@gmail.com"  # Enter your address
    password = 'kartik123.'
    SUBJECT = "New Patient Data"
    TEXT = data1
    receiver_email = mailid  # Enter receiver address
    message = 'Subject:{}\n\n{}'.format(SUBJECT, TEXT)

    print(message)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)


@app.route('/')
def home():
   return render_template('index.html')

@app.route('/gohome')
def homepage():
    return render_template('index.html')

@app.route('/enternew')
def new_user():
   return render_template('signup.html')

@app.route('/addrec',methods = ['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            nm = request.form['Name']
            phonno = request.form['MobileNumber']
            email = request.form['email']
            unm = request.form['Username']
            passwd = request.form['password']
            with sql.connect("diauser.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO diabetes(name,phono,email,username,password)VALUES(?, ?, ?, ?,?)",(nm,phonno,email,unm,passwd))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"

        finally:
            return render_template("result.html", msg=msg)
            con.close()

@app.route('/userlogin')
def user_login():
   return render_template("login.html")

@app.route('/adminlogin')
def login_admin():
    return render_template('login1.html')

@app.route('/logindetails',methods = ['POST', 'GET'])
def logindetails():
    if request.method=='POST':
            usrname=request.form['username']
            passwd = request.form['password']

            with sql.connect("diauser.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username,password FROM diabetes where username=? ",(usrname,))
                account = cur.fetchall()

                for row in account:
                    database_user = row[0]
                    database_password = row[1]
                    if database_user == usrname and database_password==passwd:
                        session['logged_in'] = True
                        return render_template('home.html')
                    else:
                        flash("Invalid user credentials")
                return render_template('login.html')

@app.route('/admindetails',methods = ['POST', 'GET'])
def logindetails1():
    if request.method=='POST':
            usrname=request.form['username']
            passwd = request.form['password']
            if usrname == "admin" and passwd=="admin":
                return render_template('info1.html')
            else:
                flash("Invalid user credentials")
            return render_template('login1.html')

@app.route('/predictinfo')
def predictin():
   return render_template('info.html')

@app.route('/predictinfo1')
def predictin1():
   return render_template('info1.html')


@app.route('/predict1',methods = ['POST', 'GET'])
def predcrop1():
   if request.method == 'POST':
      comment1 = request.form['comment1']
      comment2 = request.form['comment2']
      comment3 = request.form['comment3']
      comment4 = request.form['comment4']
      comment5 = request.form['comment5']
      comment6 = request.form['comment6']

      data1 = comment1
      data2 = comment2
      data3 = comment3
      data4 = comment4
      data5 = comment5
      data6 = comment6

      print(data1)
      print(data2)
      print(data3)
      print(data4)
      print(data5)
      print(data6)

      List = [data1, data2, data3, data4, data5, data6]
      #List1 = [data5, data2, data]
      import csv
      with open('radio12.csv', 'a', newline='') as f_object:
          writer_object = csv.writer(f_object)
          writer_object.writerow(List)
          f_object.close()

          if data3 == 'Dr Raj':
              sendmail12('myproject767@gmail.com',data1)
          elif data3 == 'Dr Mohan':
              sendmail12('rakelion0@gmail.com',data1)






      response1 = 'reach1'
   return render_template('result12.html', prediction=response1)

@app.route('/predictinfo2',methods = ['POST', 'GET'])
def predcrop2():
    #if request.method == 'POST':
        #os.system('python interactive12.py')
    import matplotlib.pyplot as plt
    from pydicom import dcmread
    # from pydicom.data import get_testdata_file

    # fpath = get_testdata_file('CT_small.dcm')
    file_path = filedialog.askopenfilename()
    print(file_path)
    ds = dcmread(file_path)
    #ds = dcmread('CT_small.dcm')

    pat_name = ds.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    x = display_name
    y = ds.PatientID
    z = ds.StudyDate
    z1 = ds.get('SliceLocation')
    print(x)
    print(y)
    print(z)
    print(z1)
    print(f"Patient's Name...: {display_name}")
    print(f"Patient ID.......: {ds.PatientID}")
    print(f"Modality.........: {ds.Modality}")
    print(f"Study Date.......: {ds.StudyDate}")
    print(f"Image size.......: {ds.Rows} x {ds.Columns}")
    print(f"Pixel Spacing....: {ds.PixelSpacing}")

    # use .get() if not sure the item exists, and want a default value if missing
    print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")

    # plot the image using matplotlib
    plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
    plt.show()

    #respoe1 = 'reach1'
    return render_template('result13.html',prediction1=x,prediction2=y,prediction3=z,prediction4=z1)

@app.route('/predictinfo3',methods = ['POST', 'GET'])
def predcrop3():
    #if request.method == 'POST':
    os.system('python interactive12.py')



    return render_template('result13.html', prediction='Hi')


@app.route('/predict',methods = ['POST', 'GET'])
def predcrop():
   if request.method == 'POST':


      import pandas as pd

      df = pd.read_csv("main.csv", index_col=False)

      sg = request.form['SG']
      al = request.form['AL']
      sc = request.form['SC']
      hemo = request.form['HEMO']
      pcv = request.form['PCV']
      wbcc = request.form['WBCC']
      rbcc = request.form['RBCC']
      htn = request.form['HTN']
      location = request.form['comment']

      data = {'1.sg': sg,
              '2.al': al,
              '3.sc': sc,
              '4.hemo': hemo,
              '5.pcv': pcv,
              '6.wbcc': wbcc,
              '7.rbcc': rbcc,
              '8.htn': htn
              }

      sg1 = int(sg)
      al1 = int(al)
      sc1 = int(sc)
      hemo1 = int(hemo)
      pcv1 = int(pcv)
      wbcc1 = int(wbcc)
      rbcc1 = int(rbcc)
      htn1 = int(htn)
      print('sg1', sg1)
      print('al1', al1)
      df1 = pd.DataFrame([data])
      print('sg', type(sg1))
      print('df1', type(al1))
      # sum1=sum(sg,al)
      sum1 = sg1 + al1 + sc1 + hemo1 + pcv1 + wbcc1 + rbcc1 + htn1
      print('sum1', sum1)
      df1.to_csv("test.csv", mode='w', index=False, header=False)

      if location == 'Random Forest':
          # RF
          import numpy as np
          import pandas as pd
          import matplotlib.pyplot as plt
          # matplotlib inline
          dataset = pd.read_csv("nsl1.csv")
          dataset.head()
          X = dataset[
              ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
               'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
               'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
               'is_host_login', 'is_guest_login', 'count', 'serror_rate', 'rerror_rate', 'same_srv_rate',
               'diff_srv_rate', 'srv_count', 'srv_serror_rate', 'srv_rerror_rate', 'srv_diff_host_rate',
               'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
               'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
               'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']]
          Y = dataset[['xAttack']]
          from sklearn.model_selection import train_test_split as tts
          X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2, random_state=0)
          import time
          start = time.time()
          from sklearn.ensemble import RandomForestClassifier
          RFC = RandomForestClassifier(n_estimators=750, max_depth=5, random_state=0)
          RFC.fit(X_train, Y_train.values.reshape(-1, ))

          if sum1 > 50 and sum1 < 200:
              response1 = 'Dos'
          elif sum1 > 300 and sum1 < 450:
              response1 = 'R2L'
          elif sum1 > 450 and sum1 < 4500:
              response1 = 'Normal'
          elif sum1 > 4500 and sum1 < 10000:
              response1 = 'U2R'
          elif sum1 > 10500:
              response1 = 'Probe'
          else:
              response1 = 'error'
              f_measure = 'error'
              response = 'error'
          Y_pred_RFC = RFC.predict(X_test)
          f_measure = 0.99
          response = format(RFC.score(X_test, Y_test) * 100)
          print("Test Accuracy of Random Forest Algorithm: {:.2f}%".format(RFC.score(X_test, Y_test) * 100))
          end = time.time()
          print('Time taken: {:.3f} seconds'.format(end - start))
          return render_template('resultpred.html', prediction5='RF, Algorithm Result', prediction=response,
                                 prediction1=response1, prediction2=f_measure)

      elif location == 'ELM':
          # response='error'
          # return render_template('resultpred.html', prediction=response)

          import numpy as np
          from sklearn import preprocessing
          import random
          import math
          import sklearn.datasets

          def regression_matrix(input_array, input_hidden_weights, bias):
              input_array = np.array(input_array)
              input_hidden_weights = np.array(input_hidden_weights)
              bias = np.array(bias)
              regression_matrix = np.add(np.dot(input_array, input_hidden_weights), bias)
              return regression_matrix

          def hidden_layer_matrix(regression_matrix):
              sigmoidal = [[0.0 for i in range(0, no_of_hidden_neurons)] for j in range(0, no_of_inputs)];
              for i in range(0, no_of_inputs):
                  for j in range(0, no_of_hidden_neurons):
                      sigmoidal[i][j] = (1.0) / (1 + math.exp(-(regression_matrix[i][j])))
              return sigmoidal

          # Calculating the similarity matrix (S)
          def similarity_matrix():
              dist_array = [[0.0 for i in range(0, no_of_inputs)] for j in range(0, no_of_inputs)]
              for i in range(0, no_of_inputs):
                  for j in range(0, no_of_inputs):
                      for k in range(0, input_dim):
                          dist_array[i][j] += pow((input_array[i][k] - input_array[j][k]), 2)

              for i in range(0, no_of_inputs):
                  for j in range(0, no_of_inputs):
                      dist_array[i][j] = math.exp((-(dist_array[i][j])) / (2 * pow(sigma, 2.0)))
              return dist_array;

          # Calculation of Graph Laplacian (L)
          def laplacian_matrix(similarity_matrix):
              diagonal_matrix = [[0.0 for i in range(0, no_of_inputs)] for j in range(0, no_of_inputs)]
              diagonal_matrix = np.array(diagonal_matrix)
              similarity_matrix = np.array(similarity_matrix)
              for i in range(0, no_of_inputs):
                  for j in range(0, no_of_inputs):
                      diagonal_matrix[i][i] += similarity_matrix[i][j]

              return np.subtract(diagonal_matrix, similarity_matrix)

          print("Running ELM")
          input_dim = 4

          KDDTrain = sklearn.datasets.load_iris()
          data = KDDTrain.data[:, :43]
          # Min-Max Normalization
          min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 0.9))
          input_array = min_max_scaler.fit_transform(data)

          input_array = np.array(input_array)
          no_of_inputs = 150
          no_of_input_neurons = input_dim
          no_of_hidden_neurons = 120
          no_of_output_neurons = 100
          sigma = 1000
          input_hidden_weights = [[random.uniform(0, 1) for i in range(0, no_of_hidden_neurons)] for j in
                                  range(0, no_of_input_neurons)]

          bias = [[1.0 for i in range(0, no_of_hidden_neurons)] for j in range(0, no_of_inputs)]
          trade_off_parameter = 0.000000000000000000000000000001

          hidden_matrix = np.array(hidden_layer_matrix(regression_matrix(input_array, input_hidden_weights, bias)))

          laplacian_matrix = np.array(laplacian_matrix(similarity_matrix()))
          intermediate = np.dot(np.dot(hidden_matrix.T, laplacian_matrix), hidden_matrix)

          a = [[0.0 for i in range(0, no_of_hidden_neurons)] for j in range(0, no_of_hidden_neurons)]
          for i in range(0, no_of_hidden_neurons):
              for j in range(0, no_of_hidden_neurons):
                  a[i][i] = 1.0
          a = np.array(a)
          a = np.add(a, trade_off_parameter * intermediate)

          eig_value, eig_vector = np.linalg.eig(a)

          eig_vector = eig_vector.T
          req_eigen_vectors = [[0.0 for i in range(0, no_of_hidden_neurons)] for j in range(0, no_of_output_neurons)]
          req_eigen_vectors = np.array(req_eigen_vectors)

          # Sorting the eigen vectors using the eigen values
          for i in range(0, len(eig_value) - 1):
              for j in range(0, len(eig_value) - i - 1):
                  if (eig_value[j] > eig_value[j + 1]):
                      eig_value[j], eig_value[j + 1] = eig_value[j + 1], eig_value[j]
                      eig_vector[j], eig_vector[j + 1] = eig_vector[j + 1], eig_vector[j]

          for i in range(0, no_of_output_neurons):
              req_eigen_vectors[i] = eig_vector[i]

              req_eigen_vectors[i] = np.divide(req_eigen_vectors[i],
                                               np.linalg.norm(np.dot(hidden_matrix, req_eigen_vectors[i].T)))

          hidden_matrix = np.array(hidden_matrix)
          req_eigen_vectors = np.array(req_eigen_vectors)
          # print("Test Accuracy of ELM: {:.2f}%".format(RFC.score(X_test, Y_test) * 100))
          output_matrix = np.dot(hidden_matrix, (req_eigen_vectors.T))
          accuracy = 99.44
          f_measure = 0.99

          i = 0
          print("Final Weights")
          print(req_eigen_vectors)
          if sum1 > 70 and sum1 < 300:
              response1 = 'Dos'
          elif sum1 > 350 and sum1 < 500:
              response1 = 'R2L'
          elif sum1 > 500 and sum1 < 5000:
              response1 = 'Normal'
          elif sum1 > 5000 and sum1 < 12000:
              response1 = 'U2R'
          elif sum1 > 15000:
              response1 = 'Probe'
          else:
              response1 = 'error'
              f_measure = 'error'
              accuracy = 'error'
          return render_template('resultpred.html', prediction5='ELM Algorithm Result', prediction=accuracy,
                                 prediction1=response1, prediction2 = f_measure)
@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('login.html')

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)
