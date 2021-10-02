# This is our Bachelor’s degree project
**The project is instructed by [prof. Lai](https://www.hsnl.cse.nsysu.edu.tw//wklai/)**

- About face-recognition and IoT setup
- tree <br />
![](/pic/tree.png)

---
##### Original idea
- Authentication by bio-feature, instead of card or something substitutable
- Automatically or manually control the electrical in room, making convinience
- Apply what we learn for now, such as database, image processing, embedding system
---
##### Usage

- Host environment setting
    ``` sudo apt update && sudo apt upgrade -y```<br />
    ``` sudo apt install docker -y ```<br />
    ``` sudo apt install docker-compose -y ```<br />

- Respberry pi 4 <br />
    - You need to compile [opecv-python](https://pimylifeup.com/raspberry-pi-opencv/) 
    - Equipped with Camera and set it up

- Run server <br />
    1\. ``` cd scc-container```<br />
    2\. ``` sudo docker-compose --env-file .env up --build```<br />

- Run Rpi client
    - [ ] it's still in develop, so now only terminal command to control demo
    - [x] command
    - ``` cd rpi_client ```
    - ```python3 run.py ```

- In Develop
    - DHT -22 detect temperature and humidity
    - Front-end better demo experience
    - Recognization rate increase