export {};
declare global {
  interface Window {
    Plaid?: {
      create(config: {
        token: string;
        onSuccess: (token: string) => void;
        onExit: () => void;
      }): {
        open(): void;
        destroy(): void;
      };
    };
  }
}
