[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

# Recommender-System

A movie recommender system using collaborative filtering, content-based filtering, and an ensemble model.

## Model Details

The model consists of three parts
- Content-based filtering
- Collaborative filtering
- Ensemble model


## Dataset 

The dataset comprises 100,000 ratings contributed by 700 individuals for a selection of 9,000 movies. It's worth noting that not all individuals have rated every movie in the dataset.

The dataset also contains information about each movie:
- `movies_metadata.csv `: contains information like genre, language, title, summary, etc.
- `ratings.csv`: contains ratings of users to movies.
- `keywords.csv`: contains keywords related to a movie.
- `credits.csv`: contains information about the cast and crew of the movies.


## Content-Based Filtering

This method relies solely on the similarity between movies and the feedback from previous behavior in order to suggest films to new members. To accomplish this, we take advantage of the features of a movie, such as its title and genre. 


## Collaborative Filtering

This method utilizes the similarities between people's tastes. The model is not aware of the actual content, which may cause some people to react similarly.


## Ensemble Model

Combining collaborative and content-based filtering, we can develop a model that recognizes both the context of the movies and the similarity between people's votes.


## MLOps
An API is provided that can be used to pass input to the model and obtain recommendations. Additionally, when the model is deployed, it is possible to monitor metrics such as precision and recall over time.

![Screenshot of MLFlow](https://github.com/aidaaminian/Recommender-System/blob/main/img1.jpg?raw=true)




## Contributing 

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

[contributors-shield]: https://img.shields.io/github/contributors/aidaaminian/Recommender-System.svg?style=for-the-badge
[contributors-url]: https://github.com/aidaaminian/Recommender-System/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/aidaaminian/Recommender-System.svg?style=for-the-badge
[forks-url]: https://github.com/aidaaminian/Recommender-System/network/members
[stars-shield]: https://img.shields.io/github/stars/aidaaminian/Recommender-System.svg?style=for-the-badge
[stars-url]: https://github.com/aidaaminian/Recommender-System/stargazers
[issues-shield]: https://img.shields.io/github/issues/aidaaminian/Recommender-System.svg?style=for-the-badge
[issues-url]: https://github.com/aidaaminian/Recommender-System/issues
[license-shield]: https://img.shields.io/github/license/aidaaminian/Recommender-System.svg?style=for-the-badge
[license-url]: https://github.com/aidaaminian/Recommender-System/blob/main/LICENSE
