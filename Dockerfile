FROM python:3.9-slim
WORKDIR /app
COPY . /app/
RUN pip install torch torchvision pandas opencv-python matplotlib numpy wandb pillow streamlit
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# Command to run the application
CMD ["streamlit", "run", "app/main.py"]

# Expose the port that Streamlit uses
EXPOSE 8501