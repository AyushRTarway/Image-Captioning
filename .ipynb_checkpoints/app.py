from flask import Flask,render_template,redirect,request

#__name__ = __main__
app = Flask(__name__)

@app.route('/')
def hello():
	return render_template("index.html")

@app.route('/',methods=["POST"])
def captioning():
	if.request.method == "POST":
		f = request.files['userfile']
		path = "C:\Users\91878\1_PYTHON_ONE\Integrating ML model with flask\Image Captioning\static\{}".format(f.filename)
		f.save(path)

	return render_template("index.html")

if __name__ == '__main__' 
	app.run(debug=True) 

