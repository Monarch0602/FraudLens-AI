
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether
)


class InvestigationPDFGenerator:

    def __init__(self):
        self.styles = getSampleStyleSheet()

        self.title_style = ParagraphStyle(
            "FraudLensTitle",
            parent=self.styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#14243A"),
            alignment=TA_CENTER,
            spaceAfter=8
        )

        self.subtitle_style = ParagraphStyle(
            "FraudLensSubtitle",
            parent=self.styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#64748B"),
            alignment=TA_CENTER,
            spaceAfter=18
        )

        self.section_style = ParagraphStyle(
            "FraudLensSection",
            parent=self.styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#1D4ED8"),
            spaceBefore=12,
            spaceAfter=7
        )

        self.body_style = ParagraphStyle(
            "FraudLensBody",
            parent=self.styles["BodyText"],
            fontSize=9,
            leading=14,
            textColor=colors.HexColor("#263445"),
            spaceAfter=5
        )

        self.small_style = ParagraphStyle(
            "FraudLensSmall",
            parent=self.styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#64748B")
        )

    @staticmethod
    def _safe_text(value):
        if value is None:
            return "N/A"

        if isinstance(value, bool):
            return "Yes" if value else "No"

        if isinstance(value, float):
            return f"{value:.4f}"

        return escape(str(value))

    def _make_paragraph(self, text, style=None):
        return Paragraph(
            self._safe_text(text),
            style or self.body_style
        )

    def _add_dictionary(self, story, data):
        rows = [["Field", "Value"]]

        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value = str(value)

            rows.append([
                self._safe_text(str(key).replace("_", " ").title()),
                self._safe_text(value)
            ])

        table = Table(
            rows,
            colWidths=[48 * mm, 112 * mm],
            repeatRows=1
        )

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))

        story.append(table)
        story.append(Spacer(1, 5 * mm))

    def _add_content(self, story, content):
        if isinstance(content, dict):
            self._add_dictionary(story, content)

        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    self._add_dictionary(story, item)
                else:
                    story.append(
                        self._make_paragraph(f"• {item}")
                    )

        else:
            story.append(self._make_paragraph(content))

    @staticmethod
    def _draw_footer(canvas, doc):
        canvas.saveState()

        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.setLineWidth(0.5)

        canvas.line(
            20 * mm,
            15 * mm,
            A4[0] - 20 * mm,
            15 * mm
        )

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))

        canvas.drawString(
            20 * mm,
            10 * mm,
            "FraudLens AI | Explainable Financial Risk Intelligence"
        )

        canvas.drawRightString(
            A4[0] - 20 * mm,
            10 * mm,
            f"Page {doc.page}"
        )

        canvas.restoreState()

    def generate_pdf(self, report):
        buffer = BytesIO()

        document = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=20 * mm,
            leftMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=22 * mm,
            title="FraudLens AI Investigation Report",
            author="FraudLens AI"
        )

        story = []

        story.append(
            Paragraph(
                "FraudLens AI",
                self.title_style
            )
        )

        story.append(
            Paragraph(
                "FINANCIAL FRAUD INVESTIGATION REPORT",
                self.subtitle_style
            )
        )

        story.append(Spacer(1, 5 * mm))

        section_order = [
            ("report_metadata", "Report Metadata"),
            ("case_information", "Case Information"),
            ("risk_assessment", "Risk Assessment"),
            ("observed_evidence", "Observed Evidence"),
            ("shap_explanations", "AI Model Explanations (SHAP)"),
            ("reasoning", "Investigation Reasoning"),
            ("recommended_actions", "Recommended Actions"),
            ("limitations", "Limitations")
        ]

        for key, title in section_order:
            if key not in report:
                continue

            story.append(
                Paragraph(
                    escape(title),
                    self.section_style
                )
            )

            self._add_content(story, report[key])

        story.append(Spacer(1, 8 * mm))

        story.append(
            Paragraph(
                "DISCLAIMER",
                self.section_style
            )
        )

        story.append(
            Paragraph(
                "This report is generated using automated risk analysis "
                "and machine-learning systems. Risk scores and model "
                "explanations are indicators for investigation and do not "
                "constitute proof of fraud. All findings must be "
                "independently verified by a qualified analyst.",
                self.body_style
            )
        )

        document.build(
            story,
            onFirstPage=self._draw_footer,
            onLaterPages=self._draw_footer
        )

        pdf_bytes = buffer.getvalue()
        buffer.close()

        return pdf_bytes
