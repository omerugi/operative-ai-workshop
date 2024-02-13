# Use the Jupyter Data Science Notebook base image
FROM jupyter/base-notebook

# Install TensorFlow, Keras, scikit-learn, Seaborn, OpenCV, NumPy, pandas, matplotlib, and PyTorch
RUN pip install tensorflow keras scikit-learn seaborn numpy pandas matplotlib faker ydata-profiling ipython ipywidgets

# Install git
USER root
RUN apt-get update && apt-get install -y git

# Set a password for the root user (consider security implications)
RUN echo "root:admin" | chpasswd

# Set up the working directory
WORKDIR /workshop

# Clone the GitHub repository
RUN git clone https://github.com/omerugi/operative-ai-workshop.git

# Change owner of the repository to jovyan
RUN chown -R jovyan:users operative-ai-workshop

# (Optional) Expose any ports needed, e.g., 8888 for Jupyter
EXPOSE 8888

# Switch back to the jovyan user
USER jovyan

# Start Jupyter Notebook
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]
