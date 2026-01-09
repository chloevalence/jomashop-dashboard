"""
AWS Lambda handler for QA Dashboard calculations.

This handler receives call data in the event payload and executes
various calculation functions from lambda_calculations module.

Event format:
{
    "action": "calculate_all_metrics" | "calculate_qa_metrics" | etc.,
    "data": [
        {
            "Call Date": "2024-01-01",
            "QA Score": 85.5,
            "Agent": "Agent 1",
            ...
        },
        ...
    ],
    "parameters": {
        # Optional parameters for specific calculations
        "threshold": 70.0,
        "lookback_days": 14,
        "days_ahead": 7,
        "current_start_date": "2024-01-01",
        "current_end_date": "2024-01-31",
        "agent": "Agent 1",
        "metric_col": "QA Score"
    }
}

Response format:
{
    "statusCode": 200,
    "body": {
        "success": true,
        "action": "calculate_all_metrics",
        "data": { ... calculated metrics ... }
    }
}
"""

import json
import logging
import traceback
from typing import Dict, Any, List, Optional

import pandas as pd
import lambda_calculations as calc

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def create_response(
    status_code: int,
    body: Dict[str, Any],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a standardized Lambda response.

    Args:
        status_code: HTTP status code
        body: Response body data
        error: Optional error message

    Returns:
        Lambda response dictionary
    """
    response_body = body.copy()
    if error:
        response_body["error"] = error
        response_body["success"] = False
    else:
        response_body["success"] = True

    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",  # Adjust CORS as needed
        },
        "body": json.dumps(response_body),
    }


def event_data_to_dataframe(event_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert event data (list of records) to pandas DataFrame.

    Args:
        event_data: List of dictionaries representing call records

    Returns:
        pandas DataFrame with proper date conversion
    """
    if not event_data:
        return pd.DataFrame()

    df = pd.DataFrame(event_data)

    # Convert Call Date to datetime if present
    if "Call Date" in df.columns:
        df["Call Date"] = pd.to_datetime(df["Call Date"], errors="coerce")

    # Ensure numeric columns are properly typed
    numeric_columns = [
        "QA Score",
        "AHT",
        "Rubric Pass Count",
        "Rubric Fail Count",
        "Revenue Retention Amount",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function for QA Dashboard calculations.

    Args:
        event: Lambda event dictionary containing action, data, and parameters
        context: Lambda context object

    Returns:
        Lambda response dictionary with calculated metrics
    """
    try:
        # Parse event body if it's a string (API Gateway format)
        if isinstance(event.get("body"), str):
            try:
                event = json.loads(event["body"])
            except json.JSONDecodeError:
                return create_response(
                    400,
                    {"message": "Invalid JSON in request body"},
                    error="Invalid JSON format",
                )

        # Extract action, data, and parameters
        action = event.get("action", "calculate_all_metrics")
        event_data = event.get("data", [])
        parameters = event.get("parameters", {})

        # Define actions that don't require data
        actions_without_data = {
            "normalize_agent_id",
            "normalize_category",
            "extract_products_from_text",
        }

        # Validate required fields (only for actions that need data)
        if action not in actions_without_data and not event_data:
            return create_response(
                400,
                {"message": "No data provided in event"},
                error="Missing 'data' field",
            )

        # Convert data to DataFrame (only if action requires it)
        df = None
        if action not in actions_without_data:
            try:
                df = event_data_to_dataframe(event_data)
            except Exception as e:
                logger.error(f"Error converting data to DataFrame: {str(e)}")
                return create_response(
                    400,
                    {"message": "Invalid data format"},
                    error=f"Data conversion error: {str(e)}",
                )

            if df.empty:
                return create_response(
                    400,
                    {"message": "DataFrame is empty after conversion"},
                    error="Empty DataFrame",
                )

            logger.info(f"Processing action: {action}, Data rows: {len(df)}")
        else:
            logger.info(f"Processing action: {action} (no data required)")

        # Route to appropriate calculation function
        result = None

        # Core metrics calculations
        if action == "calculate_all_metrics":
            result = calc.calculate_all_metrics(df)

        elif action == "calculate_qa_metrics":
            result = calc.calculate_qa_metrics(df)

        elif action == "calculate_pass_rate":
            pass_rate = calc.calculate_pass_rate(df)
            result = {"pass_rate": pass_rate}

        elif action == "calculate_pass_rate_metrics":
            result = calc.calculate_pass_rate_metrics(df)

        elif action == "calculate_aht_metrics":
            result = calc.calculate_aht_metrics(df)

        elif action == "calculate_volume_metrics":
            result = calc.calculate_volume_metrics(df)

        elif action == "calculate_rubric_metrics":
            result = calc.calculate_rubric_metrics(df)

        elif action == "calculate_agent_metrics":
            result = calc.calculate_agent_metrics(df)

        elif action == "calculate_consistency_metrics":
            result = calc.calculate_consistency_metrics(df)

        elif action == "calculate_quality_distribution":
            result = calc.calculate_quality_distribution(df)

        elif action == "calculate_trend_metrics":
            result = calc.calculate_trend_metrics(df)

        elif action == "calculate_reason_metrics":
            result = calc.calculate_reason_metrics(df)

        elif action == "calculate_outcome_metrics":
            result = calc.calculate_outcome_metrics(df)

        elif action == "calculate_product_metrics":
            result = calc.calculate_product_metrics(df)

        # Advanced analytics
        elif action == "calculate_historical_baselines":
            current_start = parameters.get("current_start_date")
            current_end = parameters.get("current_end_date")

            if not current_start or not current_end:
                return create_response(
                    400,
                    {"message": "Missing required parameters"},
                    error="current_start_date and current_end_date are required",
                )

            # Convert to datetime if strings
            if isinstance(current_start, str):
                current_start = pd.to_datetime(current_start)
            if isinstance(current_end, str):
                current_end = pd.to_datetime(current_end)

            result = calc.calculate_historical_baselines(df, current_start, current_end)

        elif action == "calculate_percentile_rankings":
            metric_col = parameters.get("metric_col", "QA Score")
            result = calc.calculate_percentile_rankings(df, metric_col)

        elif action == "calculate_trend_slope":
            if "Call Date" not in df.columns or "QA Score" not in df.columns:
                return create_response(
                    400,
                    {"message": "Missing required columns"},
                    error="Call Date and QA Score columns are required",
                )

            dates = df["Call Date"]
            scores = df["QA Score"]
            slope = calc.calculate_trend_slope(dates, scores)
            result = {"trend_slope": slope}

        elif action == "predict_future_scores":
            days_ahead = parameters.get("days_ahead", 7)
            result = calc.predict_future_scores(df, days_ahead)

        elif action == "predict_future_scores_simple":
            days_ahead = parameters.get("days_ahead", 7)
            result = calc.predict_future_scores_simple(df, days_ahead)

        elif action == "identify_at_risk_agents":
            threshold = parameters.get("threshold", 70.0)
            lookback_days = parameters.get("lookback_days", 14)
            result = calc.identify_at_risk_agents(df, threshold, lookback_days)

        elif action == "classify_trajectory":
            agent = parameters.get("agent")
            result = calc.classify_trajectory(df, agent)

        elif action == "calculate_rubric_improvements":
            previous_data = parameters.get("previous_data")
            if not previous_data:
                return create_response(
                    400,
                    {"message": "Missing required parameter"},
                    error="previous_data is required for calculate_rubric_improvements",
                )

            previous_df = event_data_to_dataframe(previous_data)
            result = calc.calculate_rubric_improvements(previous_df, df)

        # Data transformation helpers
        elif action == "normalize_agent_id":
            agent_str = parameters.get("agent_str")
            if agent_str is None:
                return create_response(
                    400,
                    {"message": "Missing required parameter"},
                    error="agent_str is required",
                )
            normalized = calc.normalize_agent_id(agent_str)
            result = {"normalized_agent_id": normalized}

        elif action == "normalize_category":
            value = parameters.get("value")
            if value is None:
                return create_response(
                    400,
                    {"message": "Missing required parameter"},
                    error="value is required",
                )
            normalized = calc.normalize_category(value)
            result = {"normalized_category": normalized}

        elif action == "extract_products_from_text":
            text = parameters.get("text", "")
            products = calc.extract_products_from_text(text)
            result = {"products": products}

        elif action == "normalize_categories_in_dataframe":
            column_name = parameters.get("column_name")
            if not column_name:
                return create_response(
                    400,
                    {"message": "Missing required parameter"},
                    error="column_name is required",
                )
            normalized_df = calc.normalize_categories_in_dataframe(df, column_name)
            # Convert DataFrame back to records for response
            result = {"normalized_data": normalized_df.to_dict("records")}

        else:
            return create_response(
                400,
                {"message": f"Unknown action: {action}"},
                error=f"Unsupported action: {action}",
            )

        # Return successful response
        response_data = {
            "action": action,
            "data": result,
        }
        if df is not None:
            response_data["row_count"] = len(df)

        return create_response(200, response_data)

    except Exception as e:
        # Log the full error for debugging
        error_trace = traceback.format_exc()
        logger.error(f"Error in lambda_handler: {str(e)}\n{error_trace}")

        return create_response(
            500,
            {
                "message": "Internal server error",
                "action": event.get("action", "unknown"),
            },
            error=str(e),
        )


# For local testing
if __name__ == "__main__":
    # Example test event
    test_event = {
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
            },
            {
                "Call Date": "2024-01-02",
                "QA Score": 90.0,
                "Agent": "Agent 2",
                "Rubric Pass Count": 6,
                "Rubric Fail Count": 0,
                "Label": "Positive",
                "Reason": "shipping status",
                "Outcome": "resolved",
                "Summary": "Customer inquired about shipping",
            },
        ],
        "parameters": {},
    }

    test_context = None

    result = lambda_handler(test_event, test_context)
    print(json.dumps(json.loads(result["body"]), indent=2))
