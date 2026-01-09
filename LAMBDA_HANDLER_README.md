# Lambda Handler for QA Dashboard Calculations

This Lambda handler provides an API for executing all calculation functions from the `lambda_calculations` module in AWS Lambda.

## Files

- `lambda_handler.py` - Main Lambda handler function
- `lambda_calculations.py` - Calculation functions module
- `lambda_requirements.txt` - Python dependencies

## Deployment

### 1. Package for Lambda

Create a deployment package:

```bash
# Install dependencies in a local directory
mkdir lambda_package
pip install -r lambda_requirements.txt -t lambda_package/

# Copy handler and calculations module
cp lambda_handler.py lambda_package/
cp lambda_calculations.py lambda_package/

# Create deployment zip
cd lambda_package
zip -r ../lambda_deployment.zip .
cd ..
```

### 2. Upload to AWS Lambda

1. Go to AWS Lambda Console
2. Create a new function (or update existing)
3. Upload the `lambda_deployment.zip` file
4. Set handler: `lambda_handler.lambda_handler`
5. Set runtime: Python 3.9 or later
6. Set timeout: 30 seconds (adjust as needed)
7. Set memory: 512 MB minimum (may need more for large datasets)

### 3. Configure Environment Variables (Optional)

- `LOG_LEVEL`: Set logging level (default: INFO)

## Usage

### Event Format

```json
{
  "action": "calculate_all_metrics",
  "data": [
    {
      "Call Date": "2024-01-01",
      "QA Score": 85.5,
      "Agent": "Agent 1",
      "Rubric Pass Count": 5,
      "Rubric Fail Count": 1,
      "Label": "Positive",
      "Reason": "order status inquiry",
      "Outcome": "resolved",
      "Summary": "Customer asked about order status",
      "AHT": 180.5
    }
  ],
  "parameters": {
    "threshold": 70.0,
    "lookback_days": 14
  }
}
```

### Response Format

```json
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*"
  },
  "body": "{
    \"success\": true,
    \"action\": \"calculate_all_metrics\",
    \"data\": { ... calculated metrics ... },
    \"row_count\": 100
  }"
}
```

## Available Actions

### Core Metrics Calculations

#### `calculate_all_metrics`
Calculate all KPIs for a dataset.

**No additional parameters required.**

**Response:** Complete metrics dictionary with all calculated KPIs.

---

#### `calculate_qa_metrics`
Calculate QA score metrics.

**No additional parameters required.**

**Response:** `{ "avg_qa_score": 87.5, "qa_score_count": 100 }`

---

#### `calculate_pass_rate`
Calculate overall pass rate.

**No additional parameters required.**

**Response:** `{ "pass_rate": 91.67 }`

---

#### `calculate_pass_rate_metrics`
Calculate detailed pass rate metrics.

**No additional parameters required.**

**Response:** `{ "pass_rate": 91.67, "pass_count": 11, "fail_count": 1 }`

---

#### `calculate_aht_metrics`
Calculate Average Handle Time metrics.

**No additional parameters required.**

**Response:** `{ "avg_aht": 180.5, "aht_count": 100 }`

---

#### `calculate_volume_metrics`
Calculate call volume metrics.

**No additional parameters required.**

**Response:** `{ "total_calls": 100, "calls_per_day": 5.0 }`

---

#### `calculate_rubric_metrics`
Calculate rubric failure metrics.

**No additional parameters required.**

**Response:** `{ "top_failure_codes": [["1.2.3", 5], ...], "total_rubric_failures": 10 }`

---

#### `calculate_agent_metrics`
Calculate agent performance distribution.

**No additional parameters required.**

**Response:** `{ "agent_performance": [{ "Agent": "Agent 1", "Avg Score": 85.5, "Call Count": 10 }], "num_agents": 5 }`

---

#### `calculate_consistency_metrics`
Calculate consistency metrics (standard deviation, etc.).

**No additional parameters required.**

**Response:** `{ "score_std": 5.2, "score_mean": 87.5, "score_median": 88.0, ... }`

---

#### `calculate_quality_distribution`
Calculate quality tier distribution.

**No additional parameters required.**

**Response:** `{ "excellent_count": 50, "good_count": 30, "fair_count": 15, "poor_count": 5, ... }`

---

#### `calculate_trend_metrics`
Calculate trend metrics (weekly/monthly aggregations).

**No additional parameters required.**

**Response:** `{ "weekly_improvement_rate": 0.5, "weekly_trend_data": {...}, "monthly_trend_data": {...} }`

---

#### `calculate_reason_metrics`
Calculate call reason distribution.

**No additional parameters required.**

**Response:** `{ "reason_distribution": {...}, "top_reason": "...", ... }`

---

#### `calculate_outcome_metrics`
Calculate call outcome distribution.

**No additional parameters required.**

**Response:** `{ "outcome_distribution": {...}, "top_outcome": "...", ... }`

---

#### `calculate_product_metrics`
Calculate product distribution from Summary/Reason/Outcome fields.

**No additional parameters required.**

**Response:** `{ "product_distribution": {...}, "top_product": "...", ... }`

---

### Advanced Analytics

#### `calculate_historical_baselines`
Calculate historical baselines for comparison.

**Required parameters:**
- `current_start_date`: Start date of current period (string or datetime)
- `current_end_date`: End date of current period (string or datetime)

**Example:**
```json
{
  "action": "calculate_historical_baselines",
  "data": [...],
  "parameters": {
    "current_start_date": "2024-01-01",
    "current_end_date": "2024-01-31"
  }
}
```

**Response:** `{ "last_30_days": {...}, "last_90_days": {...}, "year_over_year": {...} }`

---

#### `calculate_percentile_rankings`
Calculate percentile rankings for agents.

**Optional parameters:**
- `metric_col`: Column name for metric to rank (default: "QA Score")

**Example:**
```json
{
  "action": "calculate_percentile_rankings",
  "data": [...],
  "parameters": {
    "metric_col": "QA Score"
  }
}
```

**Response:** `[{ "Agent": "Agent 1", "QA Score": 85.5, "percentile": 75.0 }, ...]`

---

#### `predict_future_scores`
Predict future QA scores using time series forecasting.

**Optional parameters:**
- `days_ahead`: Number of days to forecast (default: 7)

**Example:**
```json
{
  "action": "predict_future_scores",
  "data": [...],
  "parameters": {
    "days_ahead": 7
  }
}
```

**Response:** `{ "dates": [...], "forecast": [...], "lower_bound": [...], "upper_bound": [...], "method": "prophet" }`

---

#### `predict_future_scores_simple`
Simple linear trend forecasting (fallback method).

**Optional parameters:**
- `days_ahead`: Number of days to forecast (default: 7)

---

#### `identify_at_risk_agents`
Identify agents at risk of dropping below threshold.

**Optional parameters:**
- `threshold`: QA score threshold (default: 70.0)
- `lookback_days`: Number of days to analyze (default: 14)

**Example:**
```json
{
  "action": "identify_at_risk_agents",
  "data": [...],
  "parameters": {
    "threshold": 70.0,
    "lookback_days": 14
  }
}
```

**Response:** `[{ "agent": "Agent 1", "risk_score": 75, "recent_avg": 68.5, ... }, ...]`

---

#### `classify_trajectory`
Classify agent trajectory (improving, declining, stable, volatile).

**Optional parameters:**
- `agent`: Agent ID to filter (if not provided, analyzes all agents)

**Example:**
```json
{
  "action": "classify_trajectory",
  "data": [...],
  "parameters": {
    "agent": "Agent 1"
  }
}
```

**Response:** `{ "trajectory": "improving", "slope": 0.5, "volatility": 3.2, ... }`

---

#### `calculate_rubric_improvements`
Calculate rubric code improvements by comparing failure rates.

**Required parameters:**
- `previous_data`: List of records from previous period for comparison

**Example:**
```json
{
  "action": "calculate_rubric_improvements",
  "data": [...], // Current/BPO data
  "parameters": {
    "previous_data": [...] // Previous period data
  }
}
```

**Response:** `{ "top_improvements": [...], "total_improvements": 10 }`

---

### Data Transformation Helpers

#### `normalize_agent_id`
Normalize agent ID to standard format.

**Required parameters:**
- `agent_str`: Agent ID string to normalize

**Example:**
```json
{
  "action": "normalize_agent_id",
  "data": [],
  "parameters": {
    "agent_str": "bpagent024577540"
  }
}
```

**Response:** `{ "normalized_agent_id": "Agent 2" }`

---

#### `normalize_category`
Normalize category string (reason/outcome).

**Required parameters:**
- `value`: Category string to normalize

---

#### `extract_products_from_text`
Extract products from text using keyword matching.

**Optional parameters:**
- `text`: Text string to extract products from

---

#### `normalize_categories_in_dataframe`
Normalize categories in a DataFrame column.

**Required parameters:**
- `column_name`: Name of column to normalize

**Response:** `{ "normalized_data": [...] }` (list of normalized records)

---

## Error Handling

The handler returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input, missing parameters)
- `500`: Internal Server Error

Error response format:
```json
{
  "statusCode": 400,
  "body": "{
    \"success\": false,
    \"message\": \"Missing required parameter\",
    \"error\": \"current_start_date and current_end_date are required\",
    \"action\": \"calculate_historical_baselines\"
  }"
}
```

## Example: API Gateway Integration

```python
import json
import boto3

lambda_client = boto3.client('lambda')

event = {
    "action": "calculate_all_metrics",
    "data": [
        {
            "Call Date": "2024-01-01",
            "QA Score": 85.5,
            "Agent": "Agent 1",
            "Rubric Pass Count": 5,
            "Rubric Fail Count": 1,
        }
    ],
    "parameters": {}
}

response = lambda_client.invoke(
    FunctionName='qa-dashboard-calculations',
    InvocationType='RequestResponse',
    Payload=json.dumps(event)
)

result = json.loads(response['Payload'].read())
print(result)
```

## Testing Locally

```bash
python3 lambda_handler.py
```

This will run a test with sample data and print the results.

## Performance Considerations

- **Memory**: Allocate at least 512 MB for small datasets (<1000 rows). Increase to 1024 MB or more for larger datasets.
- **Timeout**: Set to 30 seconds minimum. Increase for large datasets or complex calculations.
- **Cold Starts**: First invocation may be slower. Consider using provisioned concurrency for production.
- **Data Size**: Lambda payload limit is 6 MB (synchronous) or 256 KB (asynchronous). For larger datasets, consider using S3 to pass data references.

## Troubleshooting

1. **Import Errors**: Ensure all dependencies are included in the deployment package
2. **Timeout Errors**: Increase Lambda timeout or reduce data size
3. **Memory Errors**: Increase Lambda memory allocation
4. **Date Parsing Errors**: Ensure dates are in ISO format (YYYY-MM-DD) or can be parsed by pandas

## License

Same as the main QA Dashboard project.

