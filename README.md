# Domino Cost Dashboard (Uses Dash by Plotly)

This repo describes how to run the `Domino Cost Dashboard` app.

_Table of Contents_

- [Domino Cost Dashboard (Uses Dash by Plotly)](#domino-cost-dashboard-uses-dash-by-plotly)
- [License](#license)
- [Create a New Project](#create-a-new-project)
- [Publish the App](#publish-the-app)
  - [Checking running status (optional)](#checking-running-status-optional)
- [Accessing Domino Cost Dashboard App](#accessing-domino-cost-dashboard-app)


# License
This template is licensed under Apache 2.0 and contains the following open source components:
* dash [MIT](https://github.com/plotly/dash/blob/dev/LICENSE)
* pandas [BSD 3](https://github.com/pandas-dev/pandas/blob/main/LICENSE)

# Create a New Git Based Project

The Domino Cost Dashboard App can be launched from a git based project.

1. Go to the Projects page in Domino and click **New Project**.

   ![project's dashboard](/img/01.projectsDashboard.png)

2. Select a name and visibility for this project, then click **Next**.

   ![create new git based project](/img/02-0.createNewProject.png)

3. Select **Git Service Provider** as Hosted By

   ![create new git based project](/img/02-1.selectGitProvider.png)

5.  Select **GitHub** from Git Service Provider drop down menu

   ![create new git based project](/img/02-2.selectServiceProvider.png)

6. Select your **Git Credentials** from the drop down menu

   ![create new git based project](/img/02-3.selectGitCredentials.png)

7. Select **dominodatalab** from the Owner/Organization

   ![create new git based project](/img/02-4.selectGitOwner.png)

10. Enter **https://github.com/dominodatalab/costs-dashboard** in Repository Name field, then click **Create**

   ![create new git based project](/img/02-5.selectRepositoryName.png)


---
# Publish the App

Once a Project has been created to host the dashboard app, follow these steps to publish it:

1. Navigate to the `App` section in the sidebar of the Project menu.
 Add a title for the App that you prefer and you can easily identify. The `standard` environment and the smaller harware tier will be enought to run it.

   ![project's dashboard](/img/05.publishApp.png)

2. After publishing it, you'll be redirected to the `App Status` pages.

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
