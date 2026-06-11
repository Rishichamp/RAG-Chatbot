"""
create_sample_pdf.py
Creates a sample company handbook PDF for testing the RAG pipeline.
Output: data/sample_docs/company_handbook.pdf
"""

import os
import sys
import subprocess

try:
    from fpdf import FPDF, XPos, YPos
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF, XPos, YPos


def create_handbook():
    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "TechCorp Employee Handbook",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, "Version 3.2  |  HR Department  |  2024",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(6)

    sections = [
        (
            "1. Refund and Return Policy",
            "Customers are eligible for a full refund within 30 days of purchase, "
            "provided the product is unused and in its original packaging. After 30 days "
            "and up to 90 days, a store credit equal to the purchase price will be issued. "
            "Digital products and software licenses are non-refundable once the activation "
            "key has been used. To initiate a return, contact support@techcorp.com with "
            "your order number. Refunds are processed within 5-7 business days and will "
            "appear on your original payment method. Shipping costs are non-refundable "
            "unless the return is due to a defect or error on our part."
        ),
        (
            "2. Employee Working Hours",
            "Standard working hours are Monday to Friday, 9:00 AM to 6:00 PM in "
            "your local time zone. A one-hour lunch break is provided between 12:00 PM "
            "and 2:00 PM at the employee's discretion. Flexible work arrangements are "
            "available with manager approval, subject to a mandatory core hours window "
            "of 10:00 AM to 3:00 PM when all team members must be reachable. Remote "
            "work is permitted up to 3 days per week for employees who have completed "
            "their probationary period of 6 months. Overtime requires prior written "
            "approval from a department head and is compensated at 1.5x the hourly rate."
        ),
        (
            "3. Onboarding Process",
            "New employees follow a structured 90-day onboarding program. Week 1 "
            "covers IT setup, system access provisioning, security training, and "
            "introductory meetings with HR and the direct manager. Weeks 2 through 4 "
            "involve shadowing senior team members, product training, and completing "
            "mandatory compliance modules on the learning management system. During "
            "Month 2, employees begin independent work on assigned tasks with bi-weekly "
            "check-ins with their mentor. A formal 90-day performance review is conducted "
            "by the direct manager and HR to assess progress and set goals for the next "
            "quarter. Each new hire is assigned a buddy from Day 1 for informal guidance."
        ),
        (
            "4. Password and Security Policy",
            "All employees must use passwords of at least 12 characters containing "
            "one uppercase letter, one number, and one special character. Passwords must "
            "be changed every 90 days. Multi-factor authentication (MFA) is mandatory for "
            "all company systems and VPN access. Employees must never share passwords or "
            "use the same password across multiple systems. If you suspect your account "
            "has been compromised, report it immediately to the IT security team at "
            "security@techcorp.com. Password managers such as 1Password or Bitwarden "
            "are recommended and company-licensed versions are available through IT. "
            "Violation of this policy may result in disciplinary action."
        ),
        (
            "5. Leave and Holiday Policy",
            "Full-time employees receive 20 days of paid annual leave per year, "
            "accruing at 1.67 days per month. Unused leave of up to 10 days may be "
            "carried forward to the next calendar year. Sick leave is provided separately "
            "at 10 days per year and does not require prior approval for absences of "
            "3 days or fewer. Parental leave consists of 26 weeks at full pay for the "
            "primary caregiver and 4 weeks for the secondary caregiver, available after "
            "12 months of employment. Public holidays follow the schedule of the "
            "employee's country of residence. Leave requests must be submitted via the "
            "HR portal with a minimum of 2 weeks notice for planned absences."
        ),
        (
            "6. Product Tiers and Pricing",
            "TechCorp offers three subscription tiers. The Starter plan costs $29 "
            "per month and supports up to 5 users with basic analytics and email support. "
            "The Professional plan costs $99 per month for teams of up to 50 users, "
            "including advanced analytics, priority support, and API access. The Enterprise "
            "plan is custom priced for unlimited users and includes SSO integration, "
            "dedicated account management, a 99.9 percent uptime SLA, and custom "
            "onboarding. A 14-day free trial is available for all tiers with no credit "
            "card required. Annual billing provides a 20 percent discount on all plans. "
            "Educational institutions and non-profits qualify for a 40 percent discount."
        ),
        (
            "7. Code of Conduct",
            "All employees are expected to treat colleagues, clients, and partners "
            "with respect and professionalism. Harassment, discrimination, or bullying of "
            "any kind will not be tolerated and may result in immediate termination. "
            "Employees must disclose any conflicts of interest to their manager and HR "
            "before engaging in outside work or investments related to the company's "
            "business. Confidential company information must not be shared externally "
            "without written authorization. Social media posts that could damage the "
            "company's reputation or reveal proprietary information are prohibited. "
            "Violations of this code should be reported to hr@techcorp.com or through "
            "the anonymous ethics hotline available 24 hours a day."
        ),
    ]

    for title, content in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 9, title,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, content)
        pdf.ln(4)

    os.makedirs("data/sample_docs", exist_ok=True)
    output_path = "data/sample_docs/company_handbook.pdf"
    pdf.output(output_path)

    size = os.path.getsize(output_path)
    print(f"  ✓  Created : {output_path}")
    print(f"  ✓  Size    : {size:,} bytes")
    print(f"  ✓  Sections: {len(sections)}")


if __name__ == "__main__":
    create_handbook()