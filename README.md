# Domino Cost Dashboard (Uses Dash by Plotly)

This repo describes how to run the Domino Cost Dashboard.

_Table of Contents_

- [Create New Project (optional)](#create-new-project-optional)
- [Publish the App ](#publish-the-app)
  - [Checking running status](#checking-running-status-optional)
- [Accessing Domino Cost App](#accessing-domino-cost-app)

# Create a New Project

The Domino Cost Dashboard requires a Project to be launched from.

1. Go to the Projects page in Domino and click **New Project**.

2. Select a name and visibility for this project, then click **Create**.

---
# Publish the App 

Once a Project has been created to host the dashboard, follow these steps to publish the dashboard:

1. Download the following three files from this repository:

   - `requirements.txt`
   - `app.sh`
   - `dash-cost-dashboard.py`

2. Upload the three files to your Domino Project.

3. Click App in the Project menu. Add a title for the App. For Environment, choose the 5.10 Domino Standard Environment. For Hardware Tier, choose `Small`.

4. After publishing, you'll be redirected to the App Status page. Wait until the status changes to `Running`.


---

# Accessing Domino Cost Dashboard

Once the App's status is `Running`, you can access the App by clicking **View App**.
