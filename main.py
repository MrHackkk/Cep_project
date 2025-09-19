<<<<<<< HEAD
# main.py
from models import PPEDataCollector

if __name__ == "__main__":
    print("🏗 Construction Site PPE Detection System")
    print("="*50)

    print("1. Data Collection")
    collector = PPEDataCollector()
    collector.collect_from_camera(total_samples=80)

    # You can remove preprocessing and training code from here
=======
# main.py
from models import PPEDataCollector

if __name__ == "__main__":
    print("🏗 Construction Site PPE Detection System")
    print("="*50)

    print("1. Data Collection")
    collector = PPEDataCollector()
    collector.collect_from_camera(total_samples=80)

    # You can remove preprocessing and training code from here
>>>>>>> 63bdbc57ff2dd37cea86ebefae4c1a32377e710c
