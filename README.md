The final_model.py file contains the start of this project. At the time I knew very little about cyber security,
but was trying to practice using machine learning in python. After combining the two I came up with the idea to
try to predict WPA handshakes based on different elements. This doesn't seem all too important because it's easy
to see handshakes in a packet sniffer such as wireshark. But I wanted to see what elements made a deauth packet 
more likely to work and gather a better handshake. 

This was all part of a project in undergrad where I was also I little short on time. The pdf file in this repo
has the graphs and a little description of the project for the class. 

After the semester I came back to this project and really wanted to make it better that's when I started
building my own IDS (intrustion detection system) where I really wanted to focus on deauth attacks, since
they can easily go unnoticed. This is sill a work in progress, but hopefull will be something anyone can deploy
over their network to look for suspicious behavior. 
