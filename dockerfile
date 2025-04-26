# Use an official Python runtime as a parent image
FROM python:3.10

# Create a user to avoid running as root
RUN useradd --uid 1000 user
USER user

# Set the PATH environment variable
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the current directory contents into the container at /app
COPY --chown=user . /app

# Set the command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]
