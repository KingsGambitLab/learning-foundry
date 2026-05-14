export class HintBudget {
  private _consumed = 0;
  constructor(private readonly capacity: number) {}

  get remaining(): number {
    return Math.max(0, this.capacity - this._consumed);
  }

  get consumed(): number {
    return Math.min(this.capacity, this._consumed);
  }

  get exhausted(): boolean {
    return this.remaining === 0;
  }

  get label(): string {
    return `Hints: ${this.remaining}/${this.capacity}`;
  }

  consume(): void {
    this._consumed += 1;
  }
}
