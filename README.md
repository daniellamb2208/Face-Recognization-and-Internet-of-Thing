# This is our Bachelor’s degree project
**The project is instructed by [prof. Lai](https://www.hsnl.cse.nsysu.edu.tw//wklai/)**

- About face-recognition and IoT setup
- tree
![](/pic/tree.png)

---
##### Original idea
- Authentication by bio-feature, instead of card or something substitutable
- Automatically or manually control the electrical in room, making convinience
- Apply what we learn for now, such as database, image processing, embedding system
---
##### Usage
- Development environment
![](/pic/neofetch.png)
- Respiberry pi 
![](/pic/neofetch_rpi.png)

- Host environment setting
    ``` sudo apt update && sudo apt upgrade -y```
    ``` sudo apt install docker -y ```
    ``` sudo apt install docker-compose -y ```

- Respberry pi 4 
    - You need to compile [opecv-python](https://pimylifeup.com/raspberry-pi-opencv/) 
    - Equipped with Camera and set it up

- Run server 
    1\. ``` cd scc-container```
    2\. ``` sudo docker-compose up  --env-file .env --build```

- Run Rpi client
    - [ ] it's still in develop, so now only terminal command to control demo
    - [x] command
    - ``` cd rpi_client ```
    - ```python3 run.py ```

- In Develop
    - DHT -22 detect temperature and humidity
    - Front-end better demo experience
    - Recognization rate increase