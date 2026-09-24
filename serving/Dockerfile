FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
RUN pip install --no-cache-dir "decider-ai[serve]"
ENV DECIDER_MODEL=Mapika/decider-2b
EXPOSE 8000
CMD ["uvicorn", "decider.serve:app", "--host", "0.0.0.0", "--port", "8000"]
