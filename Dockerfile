# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --verbose -r requirements.txt

# Define environment variable
ENV PORT 8080

# Expose the port Flask will run on
EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "app.py"]
