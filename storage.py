"""
Data Storage Module
Handles saving and loading detection data using JSON files
"""

import json
import os
import csv
from datetime import datetime


class DataStorage:
    """Simple JSON-based storage for detection history"""
    
    def __init__(self, filename='detectiondata.json'):
        self.filename = filename
        self.data = []
        self.load_data()
    
    def load_data(self):
        """Load existing data from JSON file"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    try:
                        # Try loading as JSON array
                        self.data = json.load(f)
                    except:
                        # If fails, try line-by-line (JSONL format)
                        self.data = []
                        f.seek(0)
                        for line in f:
                            if line.strip():
                                try:
                                    self.data.append(json.loads(line))
                                except:
                                    continue
                print(f"Loaded {len(self.data)} previous detections")
            except Exception as e:
                print(f"Error loading data: {e}")
                self.data = []
        else:
            self.data = []
            print("Creating new data file")
    
    def save_detection(self, detection_data):
        """Save a new detection to file"""
        # Add full timestamp
        now = datetime.now()
        detection_data['datetime'] = now.strftime('%Y-%m-%d %H:%M:%S')
        detection_data['date'] = now.strftime('%Y-%m-%d')
        detection_data['time'] = now.strftime('%H:%M:%S')
        
        # Add to memory
        self.data.append(detection_data)
        
        # Save to file (append mode - JSONL format)
        try:
            with open(self.filename, 'a') as f:
                json.dump(detection_data, f)
                f.write('\n')
        except Exception as e:
            print(f"Error saving detection: {e}")
    
    def get_all_detections(self):
        """Get all stored detections"""
        return self.data
    
    def get_today_detections(self):
        """Get all detections from today"""
        today = datetime.now().strftime('%Y-%m-%d')
        return [d for d in self.data if d.get('date') == today]
    
    def get_violations(self):
        """Get all violations"""
        return [d for d in self.data if not d.get('is_compliant', False)]
    
    def get_statistics(self):
        """Calculate simple statistics"""
        total = len(self.data)
        violations = len([d for d in self.data if not d.get('is_compliant', False)])
        compliant = total - violations
        
        compliance_rate = (compliant / total * 100) if total > 0 else 0
        
        today_detections = self.get_today_detections()
        today_violations = len([d for d in today_detections if not d.get('is_compliant', False)])
        
        return {
            'total_detections': total,
            'total_violations': violations,
            'total_compliant': compliant,
            'compliance_rate': round(compliance_rate, 2),
            'today_detections': len(today_detections),
            'today_violations': today_violations
        }
    
    def export_to_csv(self, output_file='detectionsexport.csv'):
        """Export all detections to CSV file"""
        if not self.data:
            return False
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow(['Date', 'Time', 'Compliant', 'Compliance %', 'Missing Items', 'Present Items'])
                
                # Data rows
                for detection in self.data:
                    missing = ', '.join([item.get('name', '') for item in detection.get('missing_items', [])])
                    present = ', '.join([item.get('name', '') for item in detection.get('present_items', [])])
                    
                    writer.writerow([
                        detection.get('date', ''),
                        detection.get('time', ''),
                        'Yes' if detection.get('is_compliant') else 'No',
                        f"{detection.get('compliance_percentage', 0):.1f}%",
                        missing if missing else 'None',
                        present if present else 'None'
                    ])
            
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False

