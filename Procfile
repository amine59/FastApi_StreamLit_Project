web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
&
web: sh setup.sh && streamlit run StreamLit.py