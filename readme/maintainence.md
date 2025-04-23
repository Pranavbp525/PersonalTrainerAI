# Maintenance and Troubleshooting Guide

This document provides guidelines for maintaining the PersonalTrainerAI system and troubleshooting common issues.

## Routine Maintenance

### Database Maintenance

#### PostgreSQL
- **Backups**: Automated backups to GCS bucket `gs://personaltrainer-db-backups/`
- **Vacuum**: Schedule weekly vacuum analyze during low-traffic periods
- **Connection Pooling**: Monitor connection pool metrics in Cloud Monitoring

#### Vector Database (Pinecone)
- **Index Monitoring**: Check Pinecone dashboard for index health weekly
- **Data Refresh**: Update fitness knowledge base monthly with new research

### Model Monitoring

- **Agent Performance**: Review LangSmith traces weekly for errors and performance issues. Set up email alerts is error runs grow beyond 20%.
- **Evaluation**: Run `python src/chatbot/agent_eval/eval.py` bi-weekly to assess model quality

## Troubleshooting Common Issues

### Agent Response Issues

| Issue | Possible Causes | Solutions |
|-------|----------------|-----------|
| No response from agent | - PostgreSQL checkpointer error<br>- Agent crash in LangGraph | - Check PostgreSQL connection<br>- Review ELK logs for errors<br>- Restart agent container |
| Inappropriate responses | - RAG retrieval issues<br>- Prompt drift | - Check RAG evaluation scores<br>- Review and update prompts in `prompts.py` and commit them to langsmith prompt versioning|
| Slow response times | - LLM API latency<br>- DB connection issues | - Check LangSmith traces for bottlenecks<br>- Monitor DB connection pools |

### Data Pipeline Issues

| Issue | Possible Causes | Solutions |
|-------|----------------|-----------|
| Pipeline failure | - GCP permission issues<br>- Network timeouts | - Check workflow logs in Cloud Console<br>- Verify service account permissions |
| Vector DB update failure | - Embedding errors<br>- Pinecone API issues | - Check embedding model access<br>- Verify Pinecone API key validity |

### Integration Issues

| Issue | Possible Causes | Solutions |
|-------|----------------|-----------|
| Hevy API errors | - API key expired<br>- Rate limiting | - Renew API key<br>- Implement backoff strategy |
| LLM API errors | - API key issues<br>- Usage limits | - Check API key permissions<br>- Monitor usage and increase limits |

## Monitoring Alerts

The system has automated alerts configured for:
- 5xx errors exceeding threshold
- Agent evaluation score drops below 75%
- PostgreSQL high connection count
- Redis cache hit rate below 80%

Alert notifications are sent to:
- Email: pranav.bp525@gmail.com

## Deployment Rollbacks

In case of failed deployments:
1. Access Cloud Run revisions in GCP Console
2. Identify last stable revision
3. Roll back
