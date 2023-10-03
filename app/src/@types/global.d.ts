declare global {
  interface Window {
    readonly Config: {
      readonly host: string;
      readonly port: number;
    };
  }
}

export {};
