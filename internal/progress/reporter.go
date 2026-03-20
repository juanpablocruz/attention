package progress

import (
	"fmt"
	"strings"
	"time"

	"github.com/fatih/color"
	"github.com/juanpablocruz/attention/gen/internal/trainer"
)

// Reporter renders a single-line CLI progress indicator with a bar and status text.
type Reporter struct {
	epoch       int
	totalEpochs int
	total       int64
	start       time.Time
	last        time.Time
	spinIdx     int
	printed     bool
	width       int
	formatter   Formatter
}

type Formatter func(update trainer.ProgressUpdate) string

// NewReporter creates a progress reporter for one epoch.
func NewReporter(epoch, totalEpochs int, total int64, start time.Time, formatter Formatter) *Reporter {
	if formatter == nil {
		formatter = DefaultFormatter
	}
	return &Reporter{
		epoch:       epoch,
		totalEpochs: totalEpochs,
		total:       total,
		start:       start,
		last:        start,
		width:       24,
		formatter:   formatter,
	}
}

// ShouldRender reports whether enough time has passed to print another update.
func (r *Reporter) ShouldRender(now time.Time, interval time.Duration) bool {
	return now.Sub(r.last) >= interval
}

// DefaultFormatter renders a plain progress status string.
func DefaultFormatter(update trainer.ProgressUpdate) string {
	return fmt.Sprintf("loss=%.6f acc=%.4f speed=%.0f/s eta=%s", update.Loss, update.Accuracy, update.Speed, update.ETA)
}

// Render updates the progress line with the provided status data.
func (r *Reporter) Render(now time.Time, current int64, update trainer.ProgressUpdate) {
	if r.total <= 0 {
		r.total = 1
	}
	percent := float64(current) / float64(r.total)
	if percent < 0 {
		percent = 0
	}
	if percent > 1 {
		percent = 1
	}

	filled := min(int(percent*float64(r.width)), r.width)
	bar := color.MagentaString(strings.Repeat("█", filled) + strings.Repeat("░", r.width-filled))
	spinner := []rune{'⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'}
	fmt.Printf(
		"\r%s epoch=%d/%d%6.2f%% %s samples=%d/%d %s",
		color.MagentaString("%c", spinner[r.spinIdx]),
		r.epoch,
		r.totalEpochs,
		percent*100,
		bar,
		current,
		r.total,
		r.formatter(update),
	)

	r.spinIdx = (r.spinIdx + 1) % len(spinner)
	r.last = now
	r.printed = true
}

// Finish terminates the current progress line cleanly if one was rendered.
func (r *Reporter) Finish() {
	if r.printed {
		fmt.Print("\n")
	}
}
