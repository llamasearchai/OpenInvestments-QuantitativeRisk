"""
FastAPI routes for reporting endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

from ...core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ReportRequest(BaseModel):
    """Request for report generation."""
    data: List[Dict[str, Any]] = Field(..., description="Data for the report")
    report_type: str = Field(..., description="Type of report (valuation, risk, portfolio, validation)")
    format: str = Field("json", description="Report format (json, csv, html)")
    title: Optional[str] = Field(None, description="Report title")


class DashboardRequest(BaseModel):
    """Request for dashboard configuration."""
    analysis_files: List[str] = Field(..., description="List of analysis result files")
    dashboard_title: str = Field("OpenInvestments Dashboard", description="Dashboard title")


@router.post("/generate", response_model=Dict[str, Any])
async def generate_report(request: ReportRequest):
    """
    Generate comprehensive analysis reports.

    Supports multiple formats and report types.
    """
    try:
        logger.info("Received report generation request",
                   report_type=request.report_type,
                   format=request.format,
                   data_points=len(request.data))

        # Convert data to DataFrame
        df = pd.DataFrame(request.data)

        if request.format == "json":
            report_content = generate_json_report(df, request.report_type, request.title)
        elif request.format == "csv":
            report_content = generate_csv_report(df, request.report_type, request.title)
        elif request.format == "html":
            report_content = generate_html_report(df, request.report_type, request.title)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")

        result = {
            "report_metadata": {
                "title": request.title or f"{request.report_type.title()} Report",
                "type": request.report_type,
                "format": request.format,
                "generated_at": datetime.utcnow().isoformat(),
                "data_points": len(request.data)
            },
            "report_content": report_content,
            "summary": generate_summary(df, request.report_type)
        }

        logger.info("Report generation completed",
                   report_type=request.report_type,
                   format=request.format)

        return result

    except Exception as e:
        logger.error("Report generation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


def generate_json_report(df, report_type, title):
    """Generate JSON format report."""
    report_data = {
        "title": title or f"{report_type.title()} Report",
        "type": report_type,
        "generated_at": datetime.utcnow().isoformat(),
        "data_points": len(df),
        "summary": {},
        "data": df.to_dict('records')
    }

    # Calculate summary statistics
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        report_data["summary"][col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "count": int(df[col].count())
        }

    return report_data


def generate_csv_report(df, report_type, title):
    """Generate CSV format report."""
    output = StringIO()

    # Add header information
    header_info = f"""# {title or f'{report_type.title()} Report'}
# Generated at: {datetime.utcnow().isoformat()}
# Data points: {len(df)}
# Report type: {report_type}

"""

    output.write(header_info)

    # Add summary statistics
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        output.write("# Summary Statistics\n")
        summary_df = df[numeric_columns].describe()
        summary_df.to_csv(output)
        output.write("\n")

    # Add main data
    output.write("# Detailed Results\n")
    df.to_csv(output, index=False)

    return output.getvalue()


def generate_html_report(df, report_type, title):
    """Generate HTML format report."""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title or f'{report_type.title()} Report'}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
            .summary {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f8f9fa; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            .metric {{ font-weight: bold; color: #667eea; }}
            .footer {{ text-align: center; margin-top: 30px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title or f'{report_type.title()} Report'}</h1>
            <p>Report Type: {report_type}</p>
            <p>Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Data Points: {len(df)}</p>
        </div>

        <div class="summary">
            <h2>Summary Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Count</th>
                </tr>
    """

    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        html_template += ".6f"".6f"".6f"".6f""".0f""

    html_template += """
            </table>
        </div>

        <h2>Detailed Results</h2>
        <table>
            <tr>
    """

    # Add table headers
    for col in df.columns:
        html_template += f"<th>{col}</th>"

    html_template += "</tr>"

    # Add table rows (limit to first 1000 rows for performance)
    for _, row in df.head(1000).iterrows():
        html_template += "<tr>"
        for value in row:
            if isinstance(value, float):
                html_template += ".6f"
            else:
                html_template += f"<td>{value}</td>"
        html_template += "</tr>"

    if len(df) > 1000:
        html_template += f"<tr><td colspan='{len(df.columns)}' style='text-align: center; font-style: italic;'>... and {len(df) - 1000} more rows ...</td></tr>"

    html_template += """
        </table>

        <div class="footer">
            <p>Generated by OpenInvestments Quantitative Risk Platform</p>
        </div>
    </body>
    </html>
    """

    return html_template


def generate_summary(df, report_type):
    """Generate summary statistics for the report."""
    summary = {
        "data_points": len(df),
        "columns": len(df.columns),
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": len(df.select_dtypes(include=['object']).columns)
    }

    # Add report type specific summary
    if report_type == "valuation":
        if 'price' in df.columns:
            summary["average_price"] = float(df['price'].mean())
        if 'delta' in df.columns:
            summary["average_delta"] = float(df['delta'].mean())

    elif report_type == "risk":
        if 'var' in df.columns:
            summary["average_var"] = float(df['var'].mean())
        if 'es' in df.columns:
            summary["average_es"] = float(df['es'].mean())

    elif report_type == "portfolio":
        if 'return' in df.columns:
            summary["total_return"] = float(df['return'].sum())
        if 'weight' in df.columns:
            summary["assets_count"] = len(df)

    elif report_type == "validation":
        if 'mape' in df.columns:
            summary["average_mape"] = float(df['mape'].mean())
        if 'rmse' in df.columns:
            summary["average_rmse"] = float(df['rmse'].mean())

    return summary


@router.post("/dashboard", response_model=Dict[str, Any])
async def create_dashboard(request: DashboardRequest):
    """
    Create dashboard configuration from analysis results.

    Generates a dashboard configuration that can be used to visualize
    multiple analysis results in a unified interface.
    """
    try:
        logger.info("Received dashboard creation request",
                   num_files=len(request.analysis_files),
                   title=request.dashboard_title)

        dashboard_data = {
            "title": request.dashboard_title,
            "generated_at": datetime.utcnow().isoformat(),
            "sections": []
        }

        for file_path in request.analysis_files:
            try:
                # In a real implementation, this would read actual files
                # For now, we'll create mock sections based on file names
                section_type = "unknown"

                if "valuation" in file_path.lower():
                    section_type = "valuation"
                elif "risk" in file_path.lower() or "var" in file_path.lower():
                    section_type = "risk"
                elif "portfolio" in file_path.lower():
                    section_type = "portfolio"
                elif "validation" in file_path.lower():
                    section_type = "validation"

                section = {
                    "title": file_path.split('/')[-1].replace('.csv', '').replace('_', ' ').title(),
                    "type": section_type,
                    "filename": file_path,
                    "description": f"Analysis results for {section_type} metrics",
                    "visualization_type": get_visualization_type(section_type)
                }

                dashboard_data["sections"].append(section)

            except Exception as e:
                logger.warning(f"Could not process file {file_path}: {e}")
                continue

        result = {
            "dashboard_config": dashboard_data,
            "total_sections": len(dashboard_data["sections"]),
            "section_types": list(set([s["type"] for s in dashboard_data["sections"]]))
        }

        logger.info("Dashboard configuration created",
                   total_sections=len(dashboard_data["sections"]))

        return result

    except Exception as e:
        logger.error("Dashboard creation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Dashboard creation failed: {str(e)}")


def get_visualization_type(section_type):
    """Get appropriate visualization type for section."""
    visualization_map = {
        "valuation": "line_chart",
        "risk": "bar_chart",
        "portfolio": "pie_chart",
        "validation": "scatter_plot",
        "unknown": "table"
    }

    return visualization_map.get(section_type, "table")


@router.post("/export", response_model=Dict[str, Any])
async def export_data(
    data: List[Dict[str, Any]],
    format: str = "csv",
    filename: str = "export",
    include_summary: bool = True
):
    """
    Export data in various formats.

    Query parameters:
    - format: Export format (csv, json, excel)
    - filename: Output filename
    - include_summary: Whether to include summary statistics
    """
    try:
        logger.info("Received data export request",
                   format=format,
                   data_points=len(data))

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        export_result = {
            "filename": filename,
            "format": format,
            "data_points": len(data),
            "columns": list(df.columns)
        }

        if format == "csv":
            csv_output = StringIO()
            df.to_csv(csv_output, index=False)
            export_result["content"] = csv_output.getvalue()

        elif format == "json":
            export_result["content"] = df.to_dict('records')

        elif format == "excel":
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
                if include_summary:
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_columns) > 0:
                        summary_df = df[numeric_columns].describe()
                        summary_df.to_excel(writer, sheet_name='Summary')
            output.seek(0)
            import base64
            export_result["content"] = base64.b64encode(output.read()).decode('utf-8')
            export_result["content_type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        if include_summary:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            summary = {}

            for col in numeric_columns:
                summary[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }

            export_result["summary"] = summary

        export_result["timestamp"] = datetime.utcnow().isoformat()

        logger.info("Data export completed",
                   format=format,
                   data_points=len(data))

        return export_result

    except Exception as e:
        logger.error("Data export failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Data export failed: {str(e)}")
