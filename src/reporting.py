import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import datetime

class ReportGenerator:
    def __init__(self, analytics, output_dir="data"):
        self.analytics = analytics
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def export_csv(self):
        data = []
        for track_id, stats in self.analytics.stats.items():
            row = {
                "Chicken_ID": track_id,
                "Total_Distance_px": stats["total_distance"],
                "Activity_Score": stats["activity_score"],
                "Feed_Visits": stats["zone_visits"].get("Feed Zone", 0),
                "Water_Visits": stats["zone_visits"].get("Water Zone", 0),
                "Rest_Visits": stats["zone_visits"].get("Rest Zone", 0),
                "Status": "Inactive" if stats["is_inactive"] else "Active"
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"poultry_report_{timestamp}.csv")
        df.to_csv(filename, index=False)
        return filename
        
    def export_pdf(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"poultry_report_{timestamp}.pdf")
        
        c = canvas.Canvas(filename, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "Daily Poultry Observation Report")
        
        c.setFont("Helvetica", 12)
        c.drawString(100, 720, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        summary = self.analytics.get_summary_stats()
        
        y = 680
        c.drawString(100, y, f"Total Chickens Observed: {summary['total_chickens']}")
        y -= 20
        c.drawString(100, y, f"Active: {summary['active']} | Inactive: {summary['inactive']}")
        y -= 20
        c.drawString(100, y, f"Average Activity Score: {summary['avg_activity_score']}%")
        y -= 20
        c.drawString(100, y, f"Most Visited Zone: {summary['most_visited_zone']}")
        y -= 20
        c.drawString(100, y, f"Alerts Generated: {summary['alert_count']}")
        
        y -= 40
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y, "Individual Stats")
        c.setFont("Helvetica", 10)
        y -= 20
        
        for track_id, stats in list(self.analytics.stats.items())[:20]: # Limit to 20 for demo PDF
            if y < 50:
                c.showPage()
                y = 750
            status = "Inactive" if stats["is_inactive"] else "Active"
            c.drawString(100, y, f"ID: {track_id} | Score: {stats['activity_score']}% | Distance: {int(stats['total_distance'])} | Status: {status}")
            y -= 15
            
        c.save()
        return filename
