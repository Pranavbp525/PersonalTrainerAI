# Kibana Dashboard Setup for Fitness Chatbot

This guide helps you set up useful Kibana dashboards for monitoring your chatbot system.

## Initial Setup

1. Open Kibana at http://localhost:5601
2. Navigate to "Stack Management" > "Index Patterns"
3. Create a new index pattern with the pattern `fitness-chatbot-*`
4. Select `@timestamp` as the time field
5. Click "Create index pattern"

## Dashboard 1: Agent Performance Overview

1. Go to "Dashboard" and click "Create dashboard"
2. Add the following visualizations:

### Agent Routing Visualization
1. Add a new visualization > Lens
2. Configure as follows:
   - Dimensions:
     - X-axis: Terms of "selected_agent" field
     - Y-axis: Count of records
   - Title: "Agent Routing Distribution"
   - Save to dashboard

### Response Time Visualization
1. Add a new visualization > Lens
2. Configure as follows:
   - Dimensions:
     - X-axis: Date Histogram of @timestamp
     - Y-axis: Average of "duration_ms" field
     - Break down by: Terms of "agent" field
   - Title: "Response Times by Agent"
   - Save to dashboard

### Error Rate Visualization
1. Add a new visualization > Lens
2. Configure as follows:
   - Filter: level: "ERROR"
   - Dimensions:
     - X-axis: Date Histogram of @timestamp
     - Y-axis: Count of records
     - Break down by: Terms of "agent" field
   - Title: "Error Count by Agent"
   - Save to dashboard

## Dashboard 2: User Session Analysis

1. Go to "Dashboard" and click "Create dashboard"
2. Add the following visualizations:

### Session Duration Visualization
1. Add a new visualization > Lens
2. Configure as follows:
   - Dimensions:
     - X-axis: Terms of "session_id" field
     - Y-axis: Max of "total_duration_ms" field
   - Title: "Session Durations"
   - Save to dashboard

### Message Volume Visualization
1. Add a new visualization > Lens
2. Configure as follows:
   - Dimensions:
     - X-axis: Date Histogram of @timestamp
     - Y-axis: Count of records
     - Break down by: Terms of "role" field (values: "user", "assistant")
   - Title: "Message Volume Over Time"
   - Save to dashboard

## Dashboard 3: LLM Performance Metrics

1. Go to "Dashboard" and click "Create dashboard"
2. Add the following visualizations:

### LLM Response Time
1. Add a new visualization > Lens
2. Configure as follows:
   - Filter: message contains "LLM invocation"
   - Dimensions:
     - X-axis: Date Histogram of @timestamp
     - Y-axis: Average of "duration_ms" field
   - Title: "LLM Response Time"
   - Save to dashboard

### Token Usage (if available in logs)
1. Add a new visualization > Lens
2. Configure as follows:
   - Dimensions:
     - X-axis: Date Histogram of @timestamp
     - Y-axis: Sum of "prompt_tokens" field
     - Y-axis: Sum of "completion_tokens" field
   - Title: "Token Usage Over Time"
   - Save to dashboard

## Setting Up Alerts

1. Go to "Stack Management" > "Alerts and Insights" > "Rules"
2. Click "Create rule"
3. Set up the following rules:

### High Error Rate Alert
- Name: "High Error Rate Alert"
- Rule type: "Threshold"
- Index: "fitness-chatbot-*"
- Filter: level: "ERROR"
- Threshold: Count > 10 in 5 minutes
- Actions: Email notification (configure as needed)

### Slow Response Alert
- Name: "Slow Response Alert"
- Rule type: "Threshold"
- Index: "fitness-chatbot-*"
- Filter: duration_ms exists
- Threshold: Average > 5000 in 5 minutes
- Actions: Email notification (configure as needed)