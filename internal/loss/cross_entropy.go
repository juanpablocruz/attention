package loss

import (
	"math"

	"github.com/juanpablocruz/attention/gen/internal/matrix"
)

func CrossEntropyWithGrad(logits *matrix.Matrix, targets []uint8) (float32, *matrix.Matrix) {
	if logits == nil || logits.Rows == 0 || logits.Cols == 0 {
		return 0, matrix.NewZeroMatrix(0, 0)
	}

	if len(targets) != logits.Rows {
		return float32(math.NaN()), nil
	}

	probs := matrix.SoftmaxCopy(logits)
	grad := matrix.NewZeroMatrix(logits.Rows, logits.Cols)

	var totalLoss float64
	const epsilon = 1e-9

	for i := 0; i < logits.Rows; i++ {
		target := int(targets[i])
		if target < 0 || target >= logits.Cols {
			return float32(math.NaN()), nil
		}

		for j := 0; j < logits.Cols; j++ {
			grad.Vec[i][j] = probs.Vec[i][j]
		}

		p := probs.Vec[i][target]
		if p < epsilon {
			p = epsilon
		}
		totalLoss += -math.Log(float64(p))

		grad.Vec[i][target] -= 1
	}

	scale := float32(1.0 / float64(logits.Rows))
	for i := 0; i < grad.Rows; i++ {
		for j := 0; j < grad.Cols; j++ {
			grad.Vec[i][j] *= scale
		}
	}

	return float32(totalLoss) * scale, grad
}
