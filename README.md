# Domino Cost Dashboard (Uses Dash by Plotly)

This repo describes how to run the `Domino Cost Dashboard` app.

_Table of Contents_

- [Create New Project (optional)](#create-a-new-project)
- [Publish the App ](#publish-the-app)
  - [Checking running status](#checking-running-status-optional)
- [Accessing Domino Cost Dashboard App](#accessing-domino-cost-dashboard-app)

# Create a New Project

The Domino Cost Dashboard app requires a Project to be launched from.

1. Go to the Projects page in Domino and click **New Project**.

   ![project's dashboard](/img/01.projectsDashboard.png)

2. Select a name and visibility for this project, then click **Create**.

   ![create new project](/img/02.createNewProject.png)

---
# Publish the App

Once a Project has been created to host the dashboard app, follow these steps to publish it:

1. Download the following three files from this repository:

   - `requirements.txt`
   - `app.sh`
   - `dash-cost-dashboard.py`

2. Go to the `Code` section by clicking in the side bar of the project. There, upload the three files into the project.

   ![upload files](/img/03.uploadFiles.png)


3. Verify that they are uploaded properly:

   ![files in project](/img/04.files.png)

3. Navigate to the `App` section in the sidebar of the Project menu. Add a title for the App that you prefer and you can easily identify. The `standard` environment and the smaller harware tier will be enought to run it.

   ![project's dashboard](/img/05.publishApp.png)

4. After publishing it, you'll be redirected to the `App Status` pages.

   ![project's dashboard](/img/06.runApp.png)

   Wait until the status changes to `running`.

   ![project's dashboard](/img/07.appStatus.png)

   It will take a moment for the dependencies to install, and the Domino Cost App to start running.

## Checking running status (optional)
  To verify that the app is properly setup, you can check the app's user output. To access them, follow the next steps:
   1. Click on `View Execution Details` link.

   2. Click on `User Output` and you'll see a log showing the setup of the environment. Once the legend `* Running on http://127.0.0.1:8888`, your app is ready.

   ![logs](/img/08.logs.png)

---

# Accessing Domino Cost Dashboard App

Once the App's status is `Running`, you can access the App by clicking **View App**.

   ![project's dashboard](/img/07.appStatus.png)

And then Domino Cost app dashboard will be displayed

   ![project's dashboard](/img/09.dahsboard.png)
