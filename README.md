# ML-Challenge

# Machine Learning Engineer - Tasks for Applicants

## Context

Hello applicant, thank you for the great talk with my colleague. You have made a good impression on us, and we would
like to send you a technical challenge to see how you would handle such a situation. Please consider the following
challenge:

> The marketing department is doing a lot of campaigns, and they would like to know beforehand if the user segment that
> they are targeting is relevant for specific campaigns. One of the core information they need is the estimated
> customer-life-time-value of a passenger. Given that we know the activity of a passenger within the first week of
> signing up, we would like to know how, much money will they spend on trips within one month from registration.

Feel free to change or upgrade the provided stack, or to restructure the challenge's modules as you see fit. Please
provide a README file (you can rename this one as `ASSINGMENT.md`) with the explanation of your solution and
justification for the changes done and to mention any limitations you've run into. Additionally, you can also add comments and explanations where you think they might
be valuable for us to understand your implementation.

Please bundle all of your code and explanation as a zip file and send them back as described in the eMail you received.

**IMPORTANT  REMINDERS** 

This challenge is a chance for you to showcase your technical expertise, therefore try your best to make it as complete as possible with software engineering best practices, but we also understand you need to balance that with the suggested completion time of 5 days. So in case best practices are not fully implemented due to time constraints or you would implement something differently in a real production setting, we encourage you to use your README to clarify that in some way: mentioning simplifications you made, briefly describing the production-grade implementation you would opt for, ...


**Please do not publish your results.**

Let us know if you have any questions. We are also happy to get short feedback on the test itself.

Have fun!

## Data

For this we have prepared a dataset with the activity of some Free Now passengers:

passenger_activity_after_registration (stored in database.sqlite):

- *passenger_id*
- *recency_7* (1-7): high value means passenger did their last tour 7 days after registration
- *frequency_7*: number of tours done within 7 days after registration
- *monetary_value_7*: total money spent on tours within 7 days
- *frequency_30*: number of tours done within 30 days
- *monetary_value_30*: total money spent on tours within 30 days

The data in the csv looks like the following:

|  | id | recency\_7 | frequency\_7 | monetary\_value\_7 | frequency\_30 | monetary\_value\_30 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 1 | 1 | 1 | 8.5 | 1 | 8.5 |
| 1 | 2 | 1 | 1 | 16.3 | 1 | 16.3 |
| 2 | 3 | 1 | 1 | 5.6 | 1 | 5.6 |
| 3 | 4 | 1 | 1 | 32.0 | 1 | 32.0 |

You should be able to check the data using e.g. `sqlite3` similar to:
```
➜ sqlite3
SQLite version 3.32.3 2020-06-18 14:16:19
sqlite> .open database.sqlite
sqlite> .tables
passenger_activity_after_registration
sqlite> SELECT * FROM passenger_activity_after_registration limit 2;
1|1|1|8.5|1|8.5
2|1|1|16.3|1|16.3
```


## Prerequisites

* python >= 3.7
* poetry >= 1.0
* docker
* docker-compose

## Tasks

We have split up the tasks in the following 3 steps:

1. Model Training
2. Model Serving
3. Tracking

Those steps are somehow dependent on each other. They are a simplified version of potential work you might do at Free
Now. If you struggle on a certain part, feel free to simplify or abstract things.

You will already find a simple project structure as a foundation for tackling the following tasks. Feel free to adjust it the
way it fits your needs. You are also free to switch frameworks, databases or even programming languages if you feel like. The only
requirement we have, is that the process of running and testing your application is documented, reproducible and represented in the
[Makefile](./Makefile).

```shell
➜ make
help                           list available goals
run                            run application in docker (you can check api docs via: http://0.0.0.0:8080/docs)
setup                          install dependencies configured in pyproject.toml
test                           run test with docker
train                          run model training procedure (to be implemented)
```

### 1. Model Training

* Train a model to predict the money spend by a passenger in the first 30 days.
* The passenger-data for training is stored in a sqlLite
  Database (`./database.sqlite -> main.passenger_activity_after_registration`)
* You can put the training pipeline into [app/training.py](app/training.py) and execute it via `make training`. You can also
  use a Jupyter Notebook as long as you make the pipeline re-runnable and document how to do so.
* Save the model to disk. You will need it
* **Do not try to build the best model possible. Focus on a solution that generally does the job. This task should not
  consume the majority of your preparation time!**

### 2. Model Serving via HTTP

* Implement an endpoint that accepts the following JSON payload and returns the predicted monetary value based on the
  model you trained before.

```http request
POST localhost:8080/api/predict
accept: application/json
{
    "id": 1234,
    "recency_7": 1,
    "frequency_7": 1,
    "monetary_7": 8.5
}
```

* make sure you can easily replace the model in case of retraining it
* test your logic in [test/test_api.py](test/test_api.py). Feel free to add as many tests as you like

### 3. Tracking requests

* Create a table that tracks every request made to the `/predict` route
* Adjust your code so that all requests to `/predict` are saved in the newly created table
* Implement an endpoint that gets you the number of requests made per `passenger_id`.

```http request
GET localhost:8080/api/requests/1234
```
