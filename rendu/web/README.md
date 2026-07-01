# DEV WEB — Interface de chat TechCorp Assistant Financier

Interface web (Node.js/Express + HTML/JS vanilla) qui se connecte au serveur Ollama
déployé par l'équipe INFRA et permet de discuter avec le modèle `phi3.5-financial`.

## Lancer l'interface (une seule commande)

```bash
cd rendu/devweb
npm install && npm start
```

Puis ouvrir : http://localhost:3000

## Configuration

Par défaut le serveur suppose :
- Ollama sur `http://localhost:11434`
- Modèle nommé `phi3.5-financial`

Si l'équipe INFRA a créé le modèle sous un autre nom (`ollama create <nom> -f Modelfile`),
ou expose Ollama sur une autre URL, lancez avec :

```bash
OLLAMA_URL=http://<ip-infra>:11434 MODEL_NAME=<nom-du-modele> npm start
```

## Fonctionnalités

- **Statut de connexion en temps réel** : un point vert/rouge en haut à droite indique
  si le serveur devweb arrive à joindre Ollama (vérifié toutes les 5s via `/api/status`,
  qui interroge `GET /api/tags` d'Ollama). Il indique aussi si le modèle configuré
  est bien présent dans la liste des modèles disponibles sur le serveur Ollama.
- **Historique de conversation** : tous les messages (utilisateur + assistant) sont
  conservés côté client et renvoyés à chaque requête pour que le modèle garde le contexte.
- **Interface de chat** : bulles utilisateur/assistant, indicateur "réflexion en cours",
  gestion d'erreur affichée directement dans le chat si Ollama ne répond pas.

## Architecture

```
rendu/devweb/
├── server.js         # Express : sert le front + proxy vers l'API Ollama
├── package.json
└── public/
    ├── index.html
    ├── style.css
    └── script.js      # fetch /api/status, /api/chat, gestion historique
```

### Endpoints exposés par server.js

| Route          | Méthode | Rôle                                                   |
|----------------|---------|---------------------------------------------------------|
| `/api/status`  | GET     | Vérifie la connexion à Ollama (`/api/tags`)             |
| `/api/chat`    | POST    | Proxy vers `POST {OLLAMA_URL}/api/chat` (`stream:false`)|

Le proxy côté serveur évite les soucis de CORS et permet de ne jamais exposer
l'URL interne d'Ollama directement au navigateur.
