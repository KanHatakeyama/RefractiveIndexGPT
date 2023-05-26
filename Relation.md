sequenceDiagram
    participant U as User
    participant C as Client
    participant A as Authorization Server
    participant R as Resource Server

    U->>C: 1. ログイン要求
    C->>A: 2. 認可コードリクエスト
    A->>U: 3. 認可コード付きリダイレクト
    U->>C: 4. 認可コードをクライアントへ
    C->>A: 5. 認可コードとクライアント情報を送信
    A->>C: 6. アクセストークンとリフレッシュトークンを発行
    C->>R: 7. アクセストークンを使用してリソースへアクセス
    R->>C: 8. 要求されたリソースを返す