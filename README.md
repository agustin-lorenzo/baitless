# baitless
A growing set of tools for combating malicious content on the internet

<img src="bait.gif" alt="bait gif" width="500"/>

###### *This is a personal/recreational project by [Agustin Lorenzo](https://agustinlorenzo.com) - it's currently still a work in progress*

## About
Over time, content on the internet has become increasingly more negative and predatory. Social media posts have become more incentivized to farm engagement by making users angry - whether it be through everyday [ragebait](https://en.wikipedia.org/wiki/Rage-baiting) or from widespread, [purposefully controversial politics](https://www.theguardian.com/us-news/2025/nov/23/rightwing-influencers-outside-us-x-twitter-tool).

The **baitless** project is an attempt to "filter out the noise" and reduce the amount of harmful content that reaches the user. This will be done by training AI models to recognize various forms of "**bait**" that can then be highlighted, or even removed from the page entirely. Tentative plans include training models to recognize the following:

1. Logical fallacies
> This can be done by training models on publically available datasets that contain various types of fallacies

2. Propaganda
> As far as I am aware, there aren't any public datasets that contain explicit examples of propaganda. This may be detected in conjunction with fallacies - i.e. a logical fallacy in combination with nationalistic language could be an indicator of propaganda.

3. Informal "ragebait" from everyday users
> This will likely require gathering data from social media websites (e.g. Reddit or X) through webscrapping. The informal nature of the content will make it difficult to define the criteria for a post to be considered "ragebait," but they may be selected by level of controvery, number of likes vs. engagement, etc.

4. AI-generated text and videos
> It's unclear whether new models will be needed for this, since many people are already working on methods for detection AI-generated content. If performance is adequate, open source models should get the job done. Otherwise, models can be either be trained on public datasets, or on my own synthetic data.

--
## Progress
. . .
---
### To-do:
- [x] Train binary fallacy model
- [x] Train multi-label fallacy model
- [ ] Find way to detect propaganda
- [ ] Train AI-generated text detector (or check if needed)
- [ ] Train AI-generated image detector (or check if needed)
