export class HintBudget {
  private _consumed = 0;
  constructor(private readonly capacity: number) {}

  get remaining(): number {
    return this.capacity - this._consumed;
  }

  get consumed(): number {
    return this._consumed;
  }

  get exhausted(): boolean {
    return this.remaining === 0;
  }

  get label(): string {
    return `Hints: ${this.remaining}/${this.capacity}`;
  }

  consume(): void {
    if (this._consumed < this.capacity) {
      this._consumed += 1;
    }
  }
}
