import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, func
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_PATH = "webfingerprint.db"
DB_URL = f"sqlite:///{DATABASE_PATH}"

Base = declarative_base()

class Fingerprint(Base):
    __tablename__ = 'fingerprints'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    website = Column(String, nullable=False)
    website_index = Column(Integer, nullable=False)
    trace_data = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP, default=func.current_timestamp())
    
    def __repr__(self):
        return f"<Fingerprint(id={self.id}, website='{self.website}')>"

class CollectionStats(Base):
    __tablename__ = 'collection_stats'
    
    website = Column(String, primary_key=True)
    traces_collected = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<CollectionStats(website='{self.website}', traces_collected={self.traces_collected})>"

class Database:
    def __init__(self, websites):
        self.engine = create_engine(DB_URL)
        self.Session = sessionmaker(bind=self.engine)
        self.websites = websites
        
    def init_database(self):
        """Initialize the database with required tables"""
        Base.metadata.create_all(self.engine)
    
        session = self.Session()
        try:
            for website in self.websites:
                existing = session.query(CollectionStats).filter_by(website=website).first()
                if not existing:
                    session.add(CollectionStats(website=website, traces_collected=0))
            session.commit()
            print(f"Database initialized at: {os.path.abspath(DATABASE_PATH)}")
        except Exception as e:
            session.rollback()
            print(f"Error initializing database: {str(e)}")
        finally:
            session.close()
    
    def save_trace(self, website, site_idx, trace_data):
        """Save a single trace to the database"""
        trace_json = json.dumps(trace_data)
        session = self.Session()
        
        try:
            session.add(Fingerprint(
                website=website,
                website_index=site_idx,
                trace_data=trace_json
            ))
            
            stats = session.query(CollectionStats).filter_by(website=website).first()
            if stats:
                stats.traces_collected += 1
            
            session.commit()
            print(f"Saved trace for {website}")
            return True
        except Exception as e:
            session.rollback()
            print(f"Error saving trace to database: {str(e)}")
            return False
        finally:
            session.close()
    
    def insert_trace(self, website, trace_data, timestamp=None):
        """Insert a trace with simplified parameters (for collect.py compatibility)"""
        # Find website index
        try:
            site_idx = self.websites.index(website)
        except ValueError:
            print(f"Unknown website: {website}")
            return False
            
        return self.save_trace(website, site_idx, trace_data)
    
    def get_traces_collected(self):
        """Get the count of traces collected for each website"""
        session = self.Session()
        try:
            stats = session.query(CollectionStats).all()
            results = {stat.website: stat.traces_collected for stat in stats}
            return results
        except Exception as e:
            print(f"Error retrieving trace counts: {str(e)}")
            return {website: 0 for website in self.websites}
        finally:
            session.close()
    
    def export_to_json(self, output_path="dataset.json"):
        """Export the database contents to a JSON file"""
        session = self.Session()
        try:
            fingerprints = session.query(Fingerprint).order_by(Fingerprint.website, Fingerprint.id).all()
            
            # Group by website for the training script format
            websites_data = {}
            for fp in fingerprints:
                try:
                    trace_data = json.loads(fp.trace_data)
                    if fp.website not in websites_data:
                        websites_data[fp.website] = []
                    websites_data[fp.website].append({
                        'trace': trace_data,
                        'timestamp': str(fp.timestamp)
                    })
                except:
                    print(f"Error parsing trace data for {fp.website}")
            
            # Create export format
            export_data = {
                'websites': websites_data,
                'traces': [],  # For compatibility
                'metadata': {
                    'total_traces': len(fingerprints),
                    'websites_count': len(websites_data),
                    'export_format': 'database'
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            print(f"Exported {len(fingerprints)} records to {output_path}")
        except Exception as e:
            print(f"Error exporting to JSON: {str(e)}")
        finally:
            session.close()


db: Database = None
